"""
ColdCL: Training loops for all methods.
"""
import torch
import torch.nn.functional as F
from torch_geometric.utils import degree


def train_standard(model, train_data, val_data, epochs=300, lr=0.01,
                   weight_decay=5e-4, patience=50, device="cuda"):
    """Standard LP training (baseline, NodeDup)."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    best_val_auc = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        pos_edge = train_data.pos_edge_label_index
        neg_edge = train_data.neg_edge_label_index
        edge_label_index = torch.cat([pos_edge, neg_edge], dim=1)
        labels = torch.cat([
            torch.ones(pos_edge.size(1)),
            torch.zeros(neg_edge.size(1)),
        ]).to(device)

        out = model(train_data.x, train_data.edge_index, edge_label_index)
        loss = F.binary_cross_entropy_with_logits(out, labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            val_auc = _quick_eval(model, val_data, device)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience // 10:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_auc


def train_with_cl(model, train_data, val_data, cl_weight=0.5,
                  epochs=300, lr=0.01, weight_decay=5e-4, patience=50,
                  device="cuda"):
    """Training with contrastive loss (ColdCL or GlobalCL)."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    best_val_auc = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # LP loss
        pos_edge = train_data.pos_edge_label_index
        neg_edge = train_data.neg_edge_label_index
        edge_label_index = torch.cat([pos_edge, neg_edge], dim=1)
        labels = torch.cat([
            torch.ones(pos_edge.size(1)),
            torch.zeros(neg_edge.size(1)),
        ]).to(device)

        out = model(train_data.x, train_data.edge_index, edge_label_index)
        lp_loss = F.binary_cross_entropy_with_logits(out, labels)

        # CL loss
        if hasattr(model, "compute_cl_loss"):
            # ColdCL: needs num_nodes
            import inspect
            sig = inspect.signature(model.compute_cl_loss)
            if "num_nodes" in sig.parameters:
                cl_loss = model.compute_cl_loss(
                    train_data.x, train_data.edge_index, train_data.num_nodes
                )
            else:
                cl_loss = model.compute_cl_loss(
                    train_data.x, train_data.edge_index
                )
        else:
            cl_loss = torch.tensor(0.0, device=device)

        loss = lp_loss + cl_weight * cl_loss
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            val_auc = _quick_eval(model, val_data, device)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience // 10:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_auc


def train_nodedup(model, train_data, val_data, deg_threshold=5,
                  epochs=300, lr=0.01, weight_decay=5e-4, patience=50,
                  device="cuda"):
    """Training with NodeDup augmentation."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    # Augment graph once
    aug_x, aug_ei, new_num_nodes, node_map = model.augment_graph(
        train_data.x, train_data.edge_index, train_data.num_nodes, deg_threshold
    )

    best_val_auc = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        pos_edge = train_data.pos_edge_label_index
        neg_edge = train_data.neg_edge_label_index
        edge_label_index = torch.cat([pos_edge, neg_edge], dim=1)
        labels = torch.cat([
            torch.ones(pos_edge.size(1)),
            torch.zeros(neg_edge.size(1)),
        ]).to(device)

        # Encode with augmented graph
        z = model.encode(aug_x, aug_ei)
        # Decode with original node indices only
        out = model.decode(z[:train_data.num_nodes], edge_label_index)
        loss = F.binary_cross_entropy_with_logits(out, labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            val_auc = _quick_eval(model, val_data, device)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience // 10:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_auc


def train_reweight(model, train_data, val_data, deg_threshold=5, cold_weight=3.0,
                   epochs=300, lr=0.01, weight_decay=5e-4, patience=50,
                   device="cuda"):
    """
    Degree-reweighted LP training (ablation baseline).
    Upweights edges incident to cold nodes.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    # Precompute edge weights
    deg = degree(train_data.edge_index[0], num_nodes=train_data.num_nodes)
    deg = deg + degree(train_data.edge_index[1], num_nodes=train_data.num_nodes)

    pos_edge = train_data.pos_edge_label_index
    neg_edge = train_data.neg_edge_label_index
    edge_label_index = torch.cat([pos_edge, neg_edge], dim=1)

    src, dst = edge_label_index
    min_deg = torch.minimum(deg[src], deg[dst])
    weights = torch.ones(edge_label_index.size(1), device=device)
    weights[min_deg <= deg_threshold] = cold_weight

    labels = torch.cat([
        torch.ones(pos_edge.size(1)),
        torch.zeros(neg_edge.size(1)),
    ]).to(device)

    best_val_auc = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(train_data.x, train_data.edge_index, edge_label_index)
        loss = F.binary_cross_entropy_with_logits(out, labels, weight=weights)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            val_auc = _quick_eval(model, val_data, device)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience // 10:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_auc


def _quick_eval(model, data, device):
    """Quick AUC evaluation."""
    from sklearn.metrics import roc_auc_score
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        pos_edge = data.pos_edge_label_index
        neg_edge = data.neg_edge_label_index
        edge_label_index = torch.cat([pos_edge, neg_edge], dim=1)
        labels = torch.cat([
            torch.ones(pos_edge.size(1)),
            torch.zeros(neg_edge.size(1)),
        ])
        out = model(data.x, data.edge_index, edge_label_index)
        pred = torch.sigmoid(out).cpu().numpy()
        return roc_auc_score(labels.numpy(), pred)
