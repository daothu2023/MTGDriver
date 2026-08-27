"""
Utility functions for MTGDriver.

Contains:
- Graph and node feature construction from PPI_CPDB.csv.
- Driver / telomere-association label loading.
- Cross-disease pretraining label construction and pretraining routine.
- Metric and loss helpers.
- Single-epoch training and evaluation routines for the multi-task model.
"""

import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, coalesce

from model import MultiTaskGCN, LearnableAlpha

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CANCER_TYPES = [
    "BRCA", "LUAD", "CESC", "BLCA", "LIHC", "THCA",
    "ESCA", "PRAD", "STAD", "COAD", "UCEC", "LUSC",
]

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def build_cancer_data_paths(base_dir):
    return {
        cancer: {
            "features": f"{base_dir}/features_for_{cancer}.csv",
            "driver_labels": f"{base_dir}/{cancer}_labels(0_1).csv",
        }
        for cancer in CANCER_TYPES
    }

# --------------------------------------------------------------------- #
# Graph and feature construction
# --------------------------------------------------------------------- #

def load_graph_from_ppi_cpdb(ppi_path):
    """
    Build the CPDB interaction graph directly from PPI_CPDB.csv. The node
    set is the union of genes appearing in the two gene columns of this
    file (fixed, independent of the feature matrix), matching the
    manuscript's description of a single graph topology shared across
    all cancer types.
    """
    ppi_df = pd.read_csv(ppi_path)
    gene_col_a, gene_col_b = ppi_df.columns[:2]
    genes_a = ppi_df[gene_col_a].astype(str).values
    genes_b = ppi_df[gene_col_b].astype(str).values

    node_names = np.unique(np.concatenate([genes_a, genes_b]))
    num_nodes = len(node_names)
    node_to_idx = {gene: i for i, gene in enumerate(node_names)}

    print(f"[Graph] #nodes (PPI_CPDB.csv) = {num_nodes}")

    src = np.array([node_to_idx[g] for g in genes_a], dtype=np.int64)
    dst = np.array([node_to_idx[g] for g in genes_b], dtype=np.int64)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)

    print(f"[Graph] #edges (undirected, deduplicated) = {edge_index.size(1)}")
    return node_names, node_to_idx, edge_index, num_nodes


def build_node_features(features_path, node_names, node_to_idx, num_nodes, edge_index):
    """
    Map the cancer-specific feature matrix onto the fixed node set. Genes
    present in both the feature file and the graph are standardized
    (z-score); genes in the graph without a feature vector are imputed by
    averaging the (already standardized) feature vectors of their direct
    neighbors in the CPDB graph.
    """
    feat_df = pd.read_csv(features_path, index_col=0)
    feat_df.index = feat_df.index.astype(str)
    feature_dim = feat_df.shape[1]

    feature_matrix = np.zeros((num_nodes, feature_dim), dtype=np.float32)
    has_feature = np.zeros(num_nodes, dtype=bool)

    genes_with_features = feat_df.index.intersection(pd.Index(node_names))
    print(f"[Features] #genes with features (intersected with graph nodes) = {len(genes_with_features)}")

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feat_df.loc[genes_with_features].values)
    scaled_feat_df = pd.DataFrame(scaled_features, index=genes_with_features, columns=feat_df.columns)

    for gene in genes_with_features:
        idx = node_to_idx[gene]
        feature_matrix[idx] = scaled_feat_df.loc[gene].values
        has_feature[idx] = True

    neighbors = {i: [] for i in range(num_nodes)}
    for u, v in edge_index.t().tolist():
        neighbors[u].append(v)
        neighbors[v].append(u)

    for idx in range(num_nodes):
        if not has_feature[idx]:
            neighbor_features = [feature_matrix[n] for n in neighbors[idx] if has_feature[n]]
            if neighbor_features:
                feature_matrix[idx] = np.mean(neighbor_features, axis=0)

    return torch.tensor(feature_matrix, dtype=torch.float32)


def load_node_labels(driver_labels_path, telomere_labels_path, node_to_idx, num_nodes):
    driver_labels = torch.full((num_nodes,), -1, dtype=torch.long)
    telomere_labels = torch.full((num_nodes,), -1, dtype=torch.long)

    driver_df = pd.read_csv(driver_labels_path)
    telomere_df = pd.read_csv(telomere_labels_path)

    driver_df["Gene"] = driver_df["Gene"].astype(str)
    telomere_df["Gene"] = telomere_df["Gene"].astype(str)

    for gene, label in zip(driver_df["Gene"], driver_df["Labels"]):
        if gene in node_to_idx:
            driver_labels[node_to_idx[gene]] = int(label)

    for gene, label in zip(telomere_df["Gene"], telomere_df["Labels"]):
        if gene in node_to_idx:
            telomere_labels[node_to_idx[gene]] = int(label)

    return driver_labels, telomere_labels


def build_dataset_for_cancer(cancer_type, ppi_cpdb_path, telomere_labels_path, cancer_data_paths):
    node_names, node_to_idx, edge_index, num_nodes = load_graph_from_ppi_cpdb(ppi_cpdb_path)

    features_path = cancer_data_paths[cancer_type]["features"]
    driver_labels_path = cancer_data_paths[cancer_type]["driver_labels"]

    x = build_node_features(features_path, node_names, node_to_idx, num_nodes, edge_index)
    driver_labels, telomere_labels = load_node_labels(
        driver_labels_path, telomere_labels_path, node_to_idx, num_nodes
    )

    data = Data(x=x, edge_index=edge_index).to(DEVICE)
    data.y = driver_labels.to(DEVICE)
    data.y_telomere = telomere_labels.to(DEVICE)

    labeled_idx = (data.y != -1).nonzero(as_tuple=True)[0]
    print(f"[Labels] #labeled genes (driver label != -1) = {len(labeled_idx)}")
    return data, node_to_idx, labeled_idx


# --------------------------------------------------------------------- #
# Metrics and loss utilities
# --------------------------------------------------------------------- #

@torch.no_grad()
def auprc_on_mask(logits, labels, mask):
    if mask.sum().item() == 0:
        return 0.0
    masked_labels = labels[mask].detach().cpu().numpy()
    if (masked_labels == 1).sum() == 0 or (masked_labels == 0).sum() == 0:
        return 0.0
    probs = torch.sigmoid(logits[mask]).detach().cpu().numpy()
    return float(average_precision_score(masked_labels, probs))


def bce_pos_weight_from_mask(labels, mask, device):
    masked_labels = labels[mask]
    if masked_labels.numel() == 0:
        return None
    num_pos = (masked_labels == 1).sum().item()
    num_neg = (masked_labels == 0).sum().item()
    if num_pos == 0:
        return None
    return torch.tensor([num_neg / float(num_pos)], dtype=torch.float, device=device)


# --------------------------------------------------------------------- #
# Cross-disease pretraining
# --------------------------------------------------------------------- #

def build_cross_disease_pretrain_labels(node_to_idx, target_cancer, driver_labels, cancer_data_paths):
    """
    Positive: genes annotated as a cancer driver in any cancer type other
    than the target. Negative: genes never annotated as a driver in any
    cancer type. Genes with a confirmed driver/non-driver label in the
    target cancer are excluded to avoid information leakage.
    """
    num_nodes = len(node_to_idx)
    pretrain_labels = torch.full((num_nodes,), -1, dtype=torch.long)
    has_any_label = torch.zeros(num_nodes, dtype=torch.bool)

    target_labeled_mask = (driver_labels.detach().cpu() != -1)

    for cancer in CANCER_TYPES:
        if cancer == target_cancer:
            continue
        df = pd.read_csv(cancer_data_paths[cancer]["driver_labels"])
        for gene, label in zip(df["Gene"].astype(str), df["Labels"]):
            if gene in node_to_idx:
                idx = node_to_idx[gene]
                has_any_label[idx] = True
                if int(label) == 1:
                    pretrain_labels[idx] = 1

    pretrain_labels[~has_any_label] = 0
    pretrain_mask = ((pretrain_labels == 1) | (pretrain_labels == 0)) & (~target_labeled_mask)

    num_pos = int(((pretrain_labels == 1) & pretrain_mask).sum().item())
    num_neg = int(((pretrain_labels == 0) & pretrain_mask).sum().item())
    print(f"[Cross-disease pretrain labels] pos={num_pos}, neg={num_neg}")

    return pretrain_labels.to(DEVICE), pretrain_mask.to(DEVICE)


def pretrain_on_cross_disease(data, pretrain_labels, pretrain_mask, hidden_dims, dropout,
                               lr, weight_decay, max_epochs, patience):
    """
    Pretrain the encoder, shared layer, and driver head on cross-disease
    labels (loss depends only on the driver logit, so the telomere head
    receives no gradient and stays at its random initialization).
    Returns the state dict subset used to initialize the fine-tuning
    model, or None if no positive label exists for this target cancer.
    """
    model = MultiTaskGCN(in_dim=data.num_features, hidden_dims=hidden_dims, dropout=dropout).to(DEVICE)

    masked_labels = pretrain_labels[pretrain_mask]
    num_pos = int((masked_labels == 1).sum().item())
    num_neg = int((masked_labels == 0).sum().item())

    if num_pos == 0:
        print("[Cross-disease pretrain] No positive labels available; skipping pretraining.")
        return None

    pos_weight = torch.tensor([num_neg / float(num_pos)], dtype=torch.float, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        driver_logit, _ = model(data.x, data.edge_index)
        loss = criterion(driver_logit[pretrain_mask], pretrain_labels[pretrain_mask].float())
        loss.backward()
        optimizer.step()

        current_loss = float(loss.item())
        if current_loss < best_loss - 1e-6:
            best_loss = current_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"[Cross-disease pretrain] Early stop at epoch {epoch}, best loss={best_loss:.4f}")
            break

    pretrained_subset = {
        key: value for key, value in best_state.items()
        if key.startswith("encoder.") or key.startswith("shared.") or key.startswith("driver_head.")
    }
    return pretrained_subset


# --------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------- #

def train_one_epoch(model, data, train_mask, telomere_train_mask, optimizer,
                    driver_criterion, telomere_criterion, alpha_module, epoch, warmup_epochs):
    model.train()
    optimizer.zero_grad()

    driver_logit, telomere_logit = model(data.x, data.edge_index)

    driver_loss = (
        driver_criterion(driver_logit[train_mask], data.y[train_mask].float())
        if driver_criterion is not None and train_mask.sum().item() > 0
        else torch.tensor(0.0, device=driver_logit.device)
    )
    telomere_loss = (
        telomere_criterion(telomere_logit[telomere_train_mask], data.y_telomere[telomere_train_mask].float())
        if telomere_criterion is not None and telomere_train_mask.sum().item() > 0
        else torch.tensor(0.0, device=telomere_logit.device)
    )

    if epoch < warmup_epochs:
        total_loss = driver_loss
    else:
        alpha = alpha_module()
        total_loss = alpha * driver_loss + (1.0 - alpha) * telomere_loss

    total_loss.backward()
    optimizer.step()


@torch.no_grad()
def evaluate_driver_head(model, data, mask, driver_criterion):
    model.eval()
    driver_logit, _ = model(data.x, data.edge_index)
    loss = (
        driver_criterion(driver_logit[mask], data.y[mask].float())
        if driver_criterion is not None and mask.sum().item() > 0
        else torch.tensor(0.0, device=DEVICE)
    )
    auprc = auprc_on_mask(driver_logit, data.y, mask)
    return float(loss.item()), auprc


def train_single_configuration(data, train_mask, val_mask, telomere_train_mask,
                                hidden_dims, dropout, lr, weight_decay, pretrained_state,
                                max_epochs, patience, warmup_epochs):
    """
    Train one hyperparameter configuration with early stopping based on
    validation loss. Returns the validation loss and validation AUPRC at
    the best checkpoint, together with the trained model.
    """
    pos_weight_driver = bce_pos_weight_from_mask(data.y, train_mask, DEVICE)
    pos_weight_telomere = bce_pos_weight_from_mask(data.y_telomere, telomere_train_mask, DEVICE)

    driver_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_driver) if pos_weight_driver is not None else None
    telomere_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_telomere) if pos_weight_telomere is not None else None

    model = MultiTaskGCN(in_dim=data.num_features, hidden_dims=hidden_dims, dropout=dropout).to(DEVICE)
    if pretrained_state is not None:
        model.load_state_dict(pretrained_state, strict=False)

    alpha_module = LearnableAlpha(init_alpha=0.7).to(DEVICE)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(alpha_module.parameters()),
        lr=lr, weight_decay=weight_decay,
    )

    best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    best_val_loss = float("inf")
    best_val_auprc = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        train_one_epoch(
            model, data, train_mask, telomere_train_mask, optimizer,
            driver_criterion, telomere_criterion, alpha_module, epoch, warmup_epochs,
        )
        val_loss, val_auprc = evaluate_driver_head(model, data, val_mask, driver_criterion)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_val_auprc = val_auprc
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state, strict=True)
    return best_val_loss, best_val_auprc, model


def make_masks(num_nodes, train_idx, val_idx, test_idx, device):
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask
