"""
MTGDriver training pipeline.

Two-head residual GCN (cancer-driver head + telomere-association auxiliary
head) with cross-disease supervised pretraining, nested 5-fold stratified
cross-validation, and grid search over the hyperparameter space reported in
the manuscript.

Grid search
-----------
For each outer cross-validation fold, every hyperparameter combination is
trained with early stopping based on validation loss. Among all
combinations, the configuration achieving the highest validation AUPRC is
selected and retrained to obtain the final test-set score for that fold.

Usage
-----
Full manuscript protocol for one cancer type:
    python run_model.py BRCA

Run fewer repeated runs:
    python run_model.py LUAD --num_runs 3 --data_dir ./Data

Reviewer/software quick test:
    python run_model.py BRCA --quick_test

Force CPU execution:
    python run_model.py BRCA --quick_test --device cpu
"""

import os
import json
import argparse

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import utils as utils_module
from utils import (
    DEVICE,
    set_seed,
    build_cancer_data_paths,
    build_dataset_for_cancer,
    build_cross_disease_pretrain_labels,
    pretrain_on_cross_disease,
    train_single_configuration,
    auprc_on_mask,
    make_masks,
)

NUM_EPOCHS = 300
PATIENCE = 30
KFOLD = 5
WARMUP_EPOCHS = 10

PRETRAIN_LR = 1e-2
PRETRAIN_WEIGHT_DECAY = 5e-4
PRETRAIN_EPOCHS = 300
PRETRAIN_PATIENCE = 30

# Hyperparameter search space, as reported in the manuscript.
DEPTH_OPTIONS = (1, 2)
HIDDEN_OPTIONS = ((64, 64), (64, 128), (128, 64), (128, 128))
DROPOUT_OPTIONS = (0.3, 0.4, 0.5)
LR_OPTIONS = (1e-2, 3e-3, 1e-3)
WEIGHT_DECAY_OPTIONS = (1e-4, 5e-4, 1e-3)


def format_configuration(depth, hidden_dims, dropout, lr, weight_decay):
    """Return one-line text describing the model/training configuration."""
    return (
        f"GCN layers={depth}, hidden dimensions={hidden_dims}, "
        f"dropout={dropout}, learning rate={lr}, weight decay={weight_decay}"
    )


def configure_device(device_arg):
    """
    Configure the execution device.

    Parameters
    ----------
    device_arg : {"auto", "cpu", "gpu"}
        auto: use GPU if available, otherwise CPU.
        cpu: force CPU execution.
        gpu: force GPU execution; raise an error if no GPU is available.

    Notes
    -----
    PyTorch uses the "cuda" device name for both NVIDIA/CUDA and AMD/ROCm
    backends. Therefore, "gpu" maps to torch.device("cuda") when available.
    """
    global DEVICE

    if device_arg == "cpu":
        selected_device = torch.device("cpu")
    elif device_arg == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU was requested, but torch.cuda.is_available() is False.")
        selected_device = torch.device("cuda")
    else:
        selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Update both the local imported DEVICE and the DEVICE stored in utils.py,
    # because helper functions imported from utils use the module-level DEVICE.
    DEVICE = selected_device
    utils_module.DEVICE = selected_device

    return selected_device


def apply_quick_test_settings(args):
    """
    Apply reduced settings for reviewer/software testing.

    This mode keeps the full CPDB/PPI graph and real processed cancer data,
    but reduces runs, folds, epochs, and hyperparameter combinations.
    It is not intended to reproduce manuscript results.
    """
    global NUM_EPOCHS, PATIENCE, KFOLD, WARMUP_EPOCHS
    global PRETRAIN_LR, PRETRAIN_WEIGHT_DECAY, PRETRAIN_EPOCHS, PRETRAIN_PATIENCE
    global DEPTH_OPTIONS, HIDDEN_OPTIONS, DROPOUT_OPTIONS, LR_OPTIONS, WEIGHT_DECAY_OPTIONS

    print("Quick-test mode enabled")
    print("  Full CPDB/PPI graph is used.")
    print("  Reduced runs, folds, epochs, and hyperparameter search are used for software testing only.")

    args.num_runs = 1

    NUM_EPOCHS = 5
    PATIENCE = 3
    KFOLD = 2
    WARMUP_EPOCHS = 1

    PRETRAIN_LR = 1e-2
    PRETRAIN_WEIGHT_DECAY = 5e-4
    PRETRAIN_EPOCHS = 3
    PRETRAIN_PATIENCE = 2

    DEPTH_OPTIONS = (1,)
    HIDDEN_OPTIONS = ((64, 64),)
    DROPOUT_OPTIONS = (0.3,)
    LR_OPTIONS = (1e-3,)
    WEIGHT_DECAY_OPTIONS = (5e-4,)


def split_inner_train_val(outer_trainval_idx, labels, seed):
    labels_np = labels[outer_trainval_idx].detach().cpu().numpy()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_sub_idx, val_sub_idx = next(splitter.split(outer_trainval_idx.cpu(), labels_np))
    inner_train_idx = outer_trainval_idx[train_sub_idx].to(DEVICE)
    inner_val_idx = outer_trainval_idx[val_sub_idx].to(DEVICE)
    return inner_train_idx, inner_val_idx


def grid_search_for_outer_fold(data, outer_trainval_idx, outer_test_idx,
                                pretrain_labels, pretrain_mask, seed):
    """
    Split outer_trainval_idx into an inner 80/20 train/val partition,
    iterate over the hyperparameter grid, pretrain and train each
    configuration via `train_single_configuration`, and return the
    hyperparameter dictionary of the configuration with the highest
    validation AUPRC.
    """
    inner_train_idx, inner_val_idx = split_inner_train_val(outer_trainval_idx, data.y, seed)
    train_mask, val_mask, test_mask = make_masks(
        data.num_nodes, inner_train_idx, inner_val_idx, outer_test_idx.to(DEVICE), DEVICE
    )
    telomere_train_mask = (data.y_telomere != -1) & (~test_mask)

    best_hp = None
    best_val_auprc = -1.0

    total_candidates = (
        len(DEPTH_OPTIONS)
        * len(HIDDEN_OPTIONS)
        * len(DROPOUT_OPTIONS)
        * len(LR_OPTIONS)
        * len(WEIGHT_DECAY_OPTIONS)
    )
    candidate_id = 0

    for depth in DEPTH_OPTIONS:
        for hidden_pair in HIDDEN_OPTIONS:
            hidden_dims = [hidden_pair[0]] if depth == 1 else list(hidden_pair)
            for dropout in DROPOUT_OPTIONS:
                for lr in LR_OPTIONS:
                    for weight_decay in WEIGHT_DECAY_OPTIONS:
                        candidate_id += 1
                        config_text = format_configuration(depth, hidden_dims, dropout, lr, weight_decay)
                        if total_candidates == 1:
                            print(f"  Training configuration: {config_text}")
                        else:
                            print(f"  Candidate configuration {candidate_id}/{total_candidates}: {config_text}")

                        pretrained_state = pretrain_on_cross_disease(
                            data, pretrain_labels, pretrain_mask, hidden_dims, dropout,
                            PRETRAIN_LR, PRETRAIN_WEIGHT_DECAY, PRETRAIN_EPOCHS, PRETRAIN_PATIENCE,
                        )
                        val_loss, val_auprc, _ = train_single_configuration(
                            data, train_mask, val_mask, telomere_train_mask,
                            hidden_dims, dropout, lr, weight_decay, pretrained_state,
                            NUM_EPOCHS, PATIENCE, WARMUP_EPOCHS,
                        )
                        print(f"  Validation: loss={val_loss:.4f}, AUPRC={val_auprc:.4f}")

                        if val_auprc > best_val_auprc:
                            best_val_auprc = val_auprc
                            best_hp = {
                                "depth": depth,
                                "hidden_dims": hidden_dims,
                                "dropout": dropout,
                                "lr": lr,
                                "weight_decay": weight_decay,
                            }

    if total_candidates > 1:
        selected_text = format_configuration(
            best_hp["depth"], best_hp["hidden_dims"], best_hp["dropout"],
            best_hp["lr"], best_hp["weight_decay"]
        )
        print(f"  Selected configuration: {selected_text}, validation AUPRC={best_val_auprc:.4f}")

    return best_hp


def train_and_test_outer_fold(data, outer_trainval_idx, outer_test_idx,
                               pretrain_labels, pretrain_mask, best_hp, seed):
    """Retrain with the selected hyperparameters and evaluate on the
    held-out outer test set."""
    inner_train_idx, inner_val_idx = split_inner_train_val(outer_trainval_idx, data.y, seed)
    train_mask, val_mask, test_mask = make_masks(
        data.num_nodes, inner_train_idx, inner_val_idx, outer_test_idx.to(DEVICE), DEVICE
    )
    telomere_train_mask = (data.y_telomere != -1) & (~test_mask)

    pretrained_state = pretrain_on_cross_disease(
        data, pretrain_labels, pretrain_mask, best_hp["hidden_dims"], best_hp["dropout"],
        PRETRAIN_LR, PRETRAIN_WEIGHT_DECAY, PRETRAIN_EPOCHS, PRETRAIN_PATIENCE,
    )
    _, _, model = train_single_configuration(
        data, train_mask, val_mask, telomere_train_mask,
        best_hp["hidden_dims"], best_hp["dropout"], best_hp["lr"], best_hp["weight_decay"],
        pretrained_state, NUM_EPOCHS, PATIENCE, WARMUP_EPOCHS,
    )

    driver_logit, _ = model(data.x, data.edge_index)
    test_auprc = auprc_on_mask(driver_logit, data.y, test_mask)
    print(f"  Test: AUPRC={test_auprc:.4f}")
    return test_auprc


def run_nested_cv_one_seed(data, labeled_idx, pretrain_labels, pretrain_mask, seed_offset):
    """Run stratified cross-validation with grid search in each outer fold,
    for one random seed."""
    skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=41 + seed_offset)
    labels_np = data.y[labeled_idx].detach().cpu().numpy()
    fold_test_auprc = []

    for fold, (train_val_pos, test_pos) in enumerate(skf.split(labeled_idx.cpu(), labels_np), start=1):
        print("\n" + "-" * 40)
        print(f"Fold {fold}/{KFOLD}")
        outer_trainval_idx = labeled_idx[train_val_pos]
        outer_test_idx = labeled_idx[test_pos]
        assert not set(outer_trainval_idx.tolist()) & set(outer_test_idx.tolist()), "Train/test split overlap detected."

        fold_seed = 43 + seed_offset + fold
        best_hp = grid_search_for_outer_fold(
            data, outer_trainval_idx, outer_test_idx, pretrain_labels, pretrain_mask, fold_seed
        )
        test_auprc = train_and_test_outer_fold(
            data, outer_trainval_idx, outer_test_idx, pretrain_labels, pretrain_mask, best_hp, fold_seed
        )
        fold_test_auprc.append(test_auprc)
        print("-" * 40)

    print(f"\nFold summary: mean test AUPRC = {np.mean(fold_test_auprc):.4f} +/- {np.std(fold_test_auprc):.4f}")
    return fold_test_auprc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("disease", type=str, help="Target cancer type, e.g. BRCA, LUAD, UCEC")
    parser.add_argument("--data_dir", type=str, default="./Data", help="Path to the data folder")
    parser.add_argument("--results_dir", type=str, default="./results", help="Path to save results")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of repeated runs with different seeds")
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help=(
            "Run a compact software test using the full CPDB/PPI graph, one run, "
            "two folds, reduced epochs, and one fixed hyperparameter setting. "
            "This mode is not intended to reproduce manuscript results."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Execution device: auto uses GPU if available; cpu forces CPU; gpu forces GPU.",
    )
    args = parser.parse_args()

    if args.quick_test:
        apply_quick_test_settings(args)

    selected_device = configure_device(args.device)

    os.makedirs(args.results_dir, exist_ok=True)

    target_cancer = args.disease
    ppi_cpdb_path = os.path.join(args.data_dir, "PPI_CPDB.csv")
    telomere_labels_path = os.path.join(args.data_dir, "labels_telomere.csv")
    cancer_data_paths = build_cancer_data_paths(args.data_dir)

    print("Using device:", selected_device)

    run_mean_auprc = []
    all_fold_test_auprc = []

    for run in range(1, args.num_runs + 1):
        seed = 42 + run - 1
        set_seed(seed)

        print("\n" + "=" * 60)
        print(f"Run {run}/{args.num_runs} | seed={seed} | target cancer={target_cancer}")
        print("=" * 60)

        data, node_to_idx, labeled_idx = build_dataset_for_cancer(
            target_cancer, ppi_cpdb_path, telomere_labels_path, cancer_data_paths
        )
        pretrain_labels, pretrain_mask = build_cross_disease_pretrain_labels(
            node_to_idx, target_cancer, data.y, cancer_data_paths
        )

        fold_test_auprc = run_nested_cv_one_seed(data, labeled_idx, pretrain_labels, pretrain_mask, seed_offset=run)
        all_fold_test_auprc.append([float(x) for x in fold_test_auprc])
        run_mean_auprc.append(float(np.mean(fold_test_auprc)))
        print(f"\n[Run {run}/{args.num_runs}] Mean test AUPRC ({KFOLD} folds) = {run_mean_auprc[-1]:.4f}")

    print("\n" + "=" * 60)
    print(f"Summary over {args.num_runs} runs")
    print("=" * 60)
    for i, mean_auprc in enumerate(run_mean_auprc, start=1):
        print(f"  Run {i} (seed={42 + i - 1}): mean test AUPRC = {mean_auprc:.4f}")

    overall_mean = float(np.mean(run_mean_auprc))
    overall_std = float(np.std(run_mean_auprc))
    print(f"\nOverall mean test AUPRC over {len(run_mean_auprc)} runs: {overall_mean:.4f} +/- {overall_std:.4f}")

    results = {
        "target_cancer": target_cancer,
        "mode": "quick_test" if args.quick_test else "full",
        "device": str(selected_device),
        "num_runs": args.num_runs,
        "num_folds": KFOLD,
        "num_epochs": NUM_EPOCHS,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "quick_test_note": (
            "Quick-test mode uses the full CPDB/PPI graph and real processed data, "
            "but reduced runs/folds/epochs/grid search. It is intended only for "
            "software execution testing, not manuscript result reproduction."
        ) if args.quick_test else None,
        "fold_test_auprc": all_fold_test_auprc,
        "run_mean_auprc": run_mean_auprc,
        "overall_mean_auprc": overall_mean,
        "overall_std_auprc": overall_std,
    }

    prefix = "quick_test_results" if args.quick_test else "results"
    out_path = os.path.join(args.results_dir, f"{prefix}_{target_cancer}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
