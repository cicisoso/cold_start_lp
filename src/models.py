"""
ColdCL: Model definitions.

Includes:
- GCN / GraphSAGE backbones for LP
- ColdCL model with degree-gated contrastive learning
- MC Dropout variant for uncertainty estimation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import degree


# ============================================================
# Backbone encoders
# ============================================================

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, num_layers=2, dropout=0.0):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class SAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, num_layers=2, dropout=0.0):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, num_layers=2, dropout=0.0, heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


ENCODERS = {
    "GCN": GCNEncoder,
    "SAGE": SAGEEncoder,
    "GAT": GATEncoder,
}


# ============================================================
# Link predictor (dot product decoder)
# ============================================================

class LinkPredictor(nn.Module):
    """Standard LP model: encoder + dot-product decoder."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


# ============================================================
# ColdCL: Cold-Only Contrastive Learning
# ============================================================

class ColdCLModel(nn.Module):
    """
    ColdCL: Link predictor with degree-gated contrastive learning.

    Key idea: Apply augmentation + contrastive loss ONLY to low-degree
    (cold) node neighborhoods, leaving warm/hot regions untouched.
    """

    def __init__(self, encoder, hidden_channels=128, proj_dim=64,
                 deg_threshold=5, edge_drop_rate=0.3, feat_noise_std=0.1,
                 temperature=0.5):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, proj_dim),
        )
        self.deg_threshold = deg_threshold
        self.edge_drop_rate = edge_drop_rate
        self.feat_noise_std = feat_noise_std
        self.temperature = temperature

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

    def cold_augment(self, x, edge_index, num_nodes):
        """Degree-gated augmentation: only perturb cold-node ego-nets."""
        deg = degree(edge_index[0], num_nodes=num_nodes)
        deg = deg + degree(edge_index[1], num_nodes=num_nodes)
        cold_mask = deg <= self.deg_threshold

        # Edge dropout: only drop edges incident to cold nodes
        src, dst = edge_index
        incident_cold = cold_mask[src] | cold_mask[dst]
        keep_prob = torch.ones(edge_index.size(1), device=edge_index.device)
        keep_prob[incident_cold] = 1.0 - self.edge_drop_rate
        keep_mask = torch.bernoulli(keep_prob).bool()
        aug_edge_index = edge_index[:, keep_mask]

        # Feature noise: only on cold nodes
        aug_x = x.clone()
        cold_idx = torch.where(cold_mask)[0]
        if len(cold_idx) > 0:
            noise = torch.randn(len(cold_idx), x.size(1), device=x.device) * self.feat_noise_std
            aug_x[cold_idx] = aug_x[cold_idx] + noise

        return aug_x, aug_edge_index

    def contrastive_loss(self, z_orig, z_aug, cold_mask, max_nodes=512):
        """InfoNCE loss on cold nodes only."""
        h1 = self.projector(z_orig)
        h2 = self.projector(z_aug)

        # Select cold nodes
        cold_idx = torch.where(cold_mask)[0]
        if len(cold_idx) == 0:
            return torch.tensor(0.0, device=z_orig.device)

        h1 = h1[cold_idx]
        h2 = h2[cold_idx]

        # Subsample to limit memory
        if h1.size(0) > max_nodes:
            perm = torch.randperm(h1.size(0), device=h1.device)[:max_nodes]
            h1 = h1[perm]
            h2 = h2[perm]

        h1 = F.normalize(h1, dim=1)
        h2 = F.normalize(h2, dim=1)

        sim = torch.mm(h1, h2.t()) / self.temperature
        labels = torch.arange(h1.size(0), device=sim.device)
        return F.cross_entropy(sim, labels)

    def compute_cl_loss(self, x, edge_index, num_nodes):
        """Full CL computation: augment, encode, contrast."""
        z_orig = self.encode(x, edge_index)
        aug_x, aug_ei = self.cold_augment(x, edge_index, num_nodes)
        z_aug = self.encode(aug_x, aug_ei)

        deg = degree(edge_index[0], num_nodes=num_nodes)
        deg = deg + degree(edge_index[1], num_nodes=num_nodes)
        cold_mask = deg <= self.deg_threshold

        return self.contrastive_loss(z_orig, z_aug, cold_mask)


# ============================================================
# MC Dropout for uncertainty estimation
# ============================================================

class MCDropoutPredictor(nn.Module):
    """Wraps an encoder with MC dropout for uncertainty estimation."""

    def __init__(self, in_channels, hidden_channels=128, num_layers=2,
                 dropout=0.3, encoder_type="GCN"):
        super().__init__()
        EncoderClass = ENCODERS[encoder_type]
        self.encoder = EncoderClass(in_channels, hidden_channels, num_layers, dropout=dropout)
        self.mc_dropout = nn.Dropout(p=dropout)

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        z = self.mc_dropout(z)
        return self.decode(z, edge_label_index)

    @torch.no_grad()
    def mc_predict(self, x, edge_index, edge_label_index, n_samples=30):
        """MC dropout predictions: mean and std across samples."""
        self.train()  # Keep dropout active
        preds = []
        for _ in range(n_samples):
            out = self.forward(x, edge_index, edge_label_index)
            preds.append(torch.sigmoid(out))
        preds = torch.stack(preds)
        return preds.mean(dim=0), preds.std(dim=0)


# ============================================================
# Baselines
# ============================================================

class NodeDupPredictor(nn.Module):
    """
    NodeDup baseline: duplicate low-degree nodes and add self-links
    before standard LP training.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def augment_graph(self, x, edge_index, num_nodes, deg_threshold=5):
        """Duplicate low-degree nodes."""
        deg = degree(edge_index[0], num_nodes=num_nodes)
        deg = deg + degree(edge_index[1], num_nodes=num_nodes)
        cold_idx = torch.where(deg <= deg_threshold)[0]

        if len(cold_idx) == 0:
            return x, edge_index, num_nodes, torch.arange(num_nodes)

        # Duplicate features
        dup_x = x[cold_idx]
        aug_x = torch.cat([x, dup_x], dim=0)

        # Map original cold nodes to their duplicates
        dup_offset = num_nodes
        new_num_nodes = num_nodes + len(cold_idx)

        # Add edges between nodes and their duplicates
        dup_src = cold_idx
        dup_dst = torch.arange(dup_offset, new_num_nodes, device=edge_index.device)
        dup_edges = torch.stack([
            torch.cat([dup_src, dup_dst]),
            torch.cat([dup_dst, dup_src]),
        ])

        aug_edge_index = torch.cat([edge_index, dup_edges], dim=1)

        # Node mapping (for decoding, we only use original nodes)
        node_map = torch.arange(num_nodes)

        return aug_x, aug_edge_index, new_num_nodes, node_map

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        # Only use original node embeddings for decoding
        return self.decode(z, edge_label_index)


class GlobalCLModel(nn.Module):
    """Global CL baseline: augment the entire graph uniformly."""

    def __init__(self, encoder, hidden_channels=128, proj_dim=64,
                 edge_drop_rate=0.3, feat_noise_std=0.1, temperature=0.5):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, proj_dim),
        )
        self.edge_drop_rate = edge_drop_rate
        self.feat_noise_std = feat_noise_std
        self.temperature = temperature

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

    def global_augment(self, x, edge_index):
        keep_prob = 1.0 - self.edge_drop_rate
        mask = torch.bernoulli(torch.full((edge_index.size(1),), keep_prob, device=edge_index.device)).bool()
        aug_ei = edge_index[:, mask]
        aug_x = x + torch.randn_like(x) * self.feat_noise_std
        return aug_x, aug_ei

    def contrastive_loss(self, z1, z2, max_nodes=512):
        h1 = self.projector(z1)
        h2 = self.projector(z2)
        if h1.size(0) > max_nodes:
            perm = torch.randperm(h1.size(0), device=h1.device)[:max_nodes]
            h1 = h1[perm]
            h2 = h2[perm]
        h1 = F.normalize(h1, dim=1)
        h2 = F.normalize(h2, dim=1)
        sim = torch.mm(h1, h2.t()) / self.temperature
        labels = torch.arange(h1.size(0), device=sim.device)
        return F.cross_entropy(sim, labels)

    def compute_cl_loss(self, x, edge_index):
        z_orig = self.encode(x, edge_index)
        aug_x, aug_ei = self.global_augment(x, edge_index)
        z_aug = self.encode(aug_x, aug_ei)
        return self.contrastive_loss(z_orig, z_aug)
