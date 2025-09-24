# =========================================
# 1. IMPORT LIBRARIES
# =========================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# =========================================
# 2. FLUX NETWORK DEFINITION
# =========================================
class FluxNet(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x):
        return self.net(x)

# =========================================
# 3. CONTRASTIVE LOSS
# =========================================
def contrastive_loss(z1, z2, label, margin=1.0):
    cos_sim = nn.functional.cosine_similarity(z1, z2)
    pos_loss = (1 - cos_sim) * label
    neg_loss = torch.clamp(cos_sim - margin, min=0) * (1 - label)
    return (pos_loss + neg_loss).mean()

# =========================================
# 4. PAIR GENERATION
# =========================================
def generate_pairs_with_labels(X_tensor, labels):
    pair_list, label_list = [], []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] != '' and labels[j] != '':
                pair_list.append((X_tensor[i], X_tensor[j]))
                label_list.append(1 if labels[i] == labels[j] else 0)
    return pair_list, torch.tensor(label_list, dtype=torch.float32)

# =========================================
# 5. FEATURE PROCESSING
# =========================================
def preprocess_features(df, numeric_cols, categorical_cols, scaler=None, encoder=None):
    # Numeric columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    if not scaler:
        scaler = StandardScaler()
        num_features = scaler.fit_transform(df[numeric_cols])
    else:
        num_features = scaler.transform(df[numeric_cols])

    # Categorical columns
    if not encoder:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        context_features = encoder.fit_transform(df[categorical_cols])
    else:
        context_features = encoder.transform(df[categorical_cols])

    X = np.hstack([num_features, context_features])
    return X, scaler, encoder

# =========================================
# 6. MEMBERSHIP / CONCEPT FIELD
# =========================================
def compute_membership(Z, labels, type_name):
    idx = [i for i, l in enumerate(labels) if l == type_name]
    if not idx:
        return None
    centroid = Z[idx].mean(dim=0)
    sim = nn.functional.cosine_similarity(Z, centroid.unsqueeze(0))
    return sim.numpy()

# =========================================
# 7. VISUALIZATION
# =========================================
def visualize_latent_space(Z, labeled_mask):
    Z_np = Z.detach().numpy()
    Z_2d = PCA(n_components=2, random_state=42).fit_transform(Z_np)
    plt.figure(figsize=(8,6))
    plt.scatter(Z_2d[:,0], Z_2d[:,1], c=labeled_mask, cmap='coolwarm', s=50)
    plt.title("PCA projection of latent flux space (labeled vs unlabeled)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.show()

def visualize_gradients(X_tensor, feature_names):
    X_tensor.requires_grad_(True)
    Z_grad = X_tensor.model(X_tensor)
    Z_sum = Z_grad.sum(dim=1)
    Z_sum.backward(torch.ones_like(Z_sum))
    gradients = X_tensor.grad.detach().numpy()
    grad_mag = np.linalg.norm(gradients, axis=0)
    plt.figure(figsize=(10,4))
    plt.bar(range(len(grad_mag)), grad_mag)
    plt.xticks(range(len(grad_mag)), feature_names, rotation=90)
    plt.ylabel("Gradient magnitude")
    plt.title("Feature contributions to latent flux")
    plt.tight_layout()
    plt.show()
