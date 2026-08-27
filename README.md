# MTGDriver: Multitask Graph Learning for Cancer Driver Gene Prioritization

MTGDriver is a biologically informed multitask graph-learning framework
for cancer driver gene prioritization. It integrates cancer-specific
multi-omics features, KEGG pathway information, and protein-protein
interactions (PPIs) into a shared gene graph. A residual GCN encoder is
shared between two prediction heads, the cancer-driver head (primary
task) and the telomere-association head (auxiliary task), combined with
cross-disease supervised pretraining and a nested cross-validation
protocol with grid search over the hyperparameter space.

## Key Features

- **Multi-Omics and KEGG Pathway Integration:** cancer-specific node
features combine multi-omics signals (mutation frequency, copy-number
alteration, promoter methylation, differential gene expression) with
KEGG pathway membership indicators, integrated over a fixed
protein-protein interaction (PPI) network from CPDB.
- **Residual GCN Encoder:** stacks GCN layers with a residual connection
from the input node features to the final node embedding.
- **Multi-Task Heads:** a shared MLP layer feeds two linear output heads,
the cancer-driver head and the telomere-association head, combined via
a learnable weighting coefficient (alpha).
- **Cross-Disease Supervised Pretraining:** pretrains the encoder on
driver-gene labels aggregated from other cancer types before
fine-tuning on the target cancer.
- **Nested Cross-Validation with Grid Search:** 5-fold outer
cross-validation, each fold preceded by a grid search over GCN depth,
hidden dimensions, dropout rate, learning rate, and weight decay.
Each configuration is trained with early stopping on validation loss;
the configuration with the highest validation AUPRC is retrained to
obtain the held-out test score for that fold.


## Requirements

- Python 3.9+
- torch >= 1.9.1
- torch-geometric >= 2.0.4
- numpy >= 1.21.5
- pandas >= 1.3.5
- scikit-learn >= 1.0.2

Install with:

```
pip install -r requirements.txt
```

## Data

All required data files are included in the `Data/` folder of this
repository, so no manual download is needed; just clone the repository
and run.

- `PPI_CPDB.csv`: protein-protein interaction edge list (two gene-name columns)
- `features_for_<CANCER>.csv`: node feature matrix for each cancer type
(genes as index), combining cancer-specific multi-omics features
(mutation frequency, copy-number alteration, promoter methylation,
differential gene expression) with KEGG pathway membership indicators
- `<CANCER>_labels(0_1).csv`: driver-gene labels for each cancer type, columns `Gene,Labels`
- `labels_telomere.csv`: auxiliary task labels, columns `Gene,Labels`

`<CANCER>` must match one of: `BRCA, LUAD, CESC, BLCA, LIHC, THCA, ESCA, PRAD, STAD, COAD, UCEC, LUSC`.

## Usage

```
# Run MTGDriver for BRCA with default settings (10 runs x 5-fold nested CV with grid search)
python run_model.py BRCA

# Run for LUAD with custom data/results folders and fewer runs
python run_model.py LUAD --data_dir ./Data --results_dir ./results --num_runs 3

# Quick test to verify the pipeline works end-to-end (reduced runs, folds, epochs, and grid search)
python run_model.py BRCA --quick_test --device cpu
```

The script will:

- Load the CPDB protein-protein interaction graph and the cancer-specific
node feature matrix, imputing missing features by neighbor averaging.
- Load the driver-gene labels for the target cancer and the
telomere-association labels.
- Build cross-disease pretraining labels from all other cancer types.
- For each random seed, run nested 5-fold cross-validation: for each
outer fold, perform a grid search over the hyperparameter space
(early stopping on validation loss per configuration, configuration
selection by validation AUPRC), then retrain with the selected
configuration and evaluate on the held-out test fold.
- Save per-run mean test AUPRC scores to `results/results_<CANCER>.json` and print a summary across all runs.

> **Note:** The `--quick_test` flag is for quick verification only. It uses a single fixed hyperparameter
> configuration, 1 run, 2 folds, and reduced epochs, so results are not representative of full model performance.

### Optional arguments

| Argument        | Default     | Description                                                                  |
| --------------- | ----------- | ---------------------------------------------------------------------------- |
| `--data_dir`    | `./Data`    | Path to the data folder                                                      |
| `--results_dir` | `./results` | Path to save results                                                         |
| `--num_runs`    | `10`        | Number of repeated runs with different seeds                                 |
| `--quick_test`  | `False`     | Enable quick-test mode: 1 run, 2 folds, 50 epochs, 1 hyperparameter config  |
| `--device`      | `auto`      | Device to use: `cpu` or `cuda` (default: use CUDA if available)              |

## Project Structure

```
MTGDriver/
├── README.md
├── requirements.txt
├── model.py        # model definitions (residual GCN encoder, multi-task heads, learnable alpha)
├── utils.py        # data loading, cross-disease pretraining, training/evaluation utilities
├── run_model.py    # main entry point (argparse, grid search, nested cross-validation)
├── Data/           # input data (included in this repository)
└── results/        # output files (evaluation results and gene rankings)
```
