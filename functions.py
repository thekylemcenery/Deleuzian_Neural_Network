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
        # Simple feedforward network: input → hidden layer → latent embedding
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),  # project input features to 64 dims
            nn.ReLU(),                 # non-linear activation
            nn.Linear(64, latent_dim)  # project to latent flux space
        )

    def forward(self, x):
        # Forward pass: map input features into latent space
        return self.net(x)

# =========================================
# 3. CONTRASTIVE LOSS
# =========================================
def contrastive_loss(z1, z2, label, margin=1.0):
    """
    Compute the contrastive loss between two embeddings. 
    This loss encourages embeddings of similar items to be close
    and embeddings of dissimilar items to be separated by at least `margin`.

    Args:
        z1 (torch.Tensor): Latent embedding of the first sample, shape (1, latent_dim) or (batch_size, latent_dim).
        z2 (torch.Tensor): Latent embedding of the second sample, same shape as z1.
        label (torch.Tensor): Binary label indicating similarity:
                              1 if the pair is similar (same class), 
                              0 if the pair is dissimilar (different class).
        margin (float, optional): Minimum distance negative pairs should maintain.
                                  Defaults to 1.0.

    Returns:
        torch.Tensor: Scalar tensor representing the mean contrastive loss over the pair(s).
    """
    # Compute cosine similarity between two embeddings
    cos_sim = nn.functional.cosine_similarity(z1, z2)
    # If same label: minimize (1 - similarity)
    pos_loss = (1 - cos_sim) * label
    # If different label: penalize similarity beyond margin
    neg_loss = torch.clamp(cos_sim - margin, min=0) * (1 - label)
    # Average the combined loss
    return (pos_loss + neg_loss).mean()

# =========================================
# 4. PAIR GENERATION
# =========================================
def generate_pairs_with_labels(X_tensor, labels):
    """
    Generate all unique pairs of examples and assign similarity labels for contrastive learning.

    This function creates training pairs for the network. Each pair is labeled 
    according to whether the two examples belong to the same category (1) or 
    different categories (0). Unlabeled entries are skipped.

    Args:
        X_tensor (torch.Tensor): Tensor of input feature vectors, shape (num_samples, num_features).
        labels (array-like): Array of category labels for each sample. Empty strings indicate unlabeled examples.

    Returns:
        tuple:
            - pair_list (list of tuple): List of pairs of feature vectors (z1, z2).
            - label_list (torch.Tensor): Tensor of binary labels for each pair (1=same class, 0=different class).
    """
    # Generate all unique pairs of examples and assign label:
    # 1 if same category, 0 if different
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
    """
   Preprocess numeric and categorical features for neural network input.

   Args:
       df (pandas.DataFrame): Raw input data containing both numeric and categorical columns.
       numeric_cols (list of str): Names of numeric columns to standardize.
       categorical_cols (list of str): Names of categorical columns to one-hot encode.
       scaler (sklearn.preprocessing.StandardScaler, optional): Fitted scaler to reuse. If None, a new one is fitted.
       encoder (sklearn.preprocessing.OneHotEncoder, optional): Fitted encoder to reuse. If None, a new one is fitted.

   Returns:
       tuple:
           - X (np.ndarray): Preprocessed feature matrix (numeric + one-hot categorical features).
           - scaler (StandardScaler): Fitted scaler used for numeric columns.
           - encoder (OneHotEncoder): Fitted encoder used for categorical columns.
   """
    # --- Numeric preprocessing ---
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')   # force numeric
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())  # fill missing
    if not scaler:
        scaler = StandardScaler()                          # fit new scaler
        num_features = scaler.fit_transform(df[numeric_cols])
    else:
        num_features = scaler.transform(df[numeric_cols])  # reuse scaler

    # --- Categorical preprocessing ---
    if not encoder:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        context_features = encoder.fit_transform(df[categorical_cols])
    else:
        context_features = encoder.transform(df[categorical_cols])

    # Combine numeric + categorical into final feature matrix
    X = np.hstack([num_features, context_features])
    return X, scaler, encoder

# =========================================
# 6. MEMBERSHIP / CONCEPT FIELD
# =========================================
def compute_membership(Z, labels, type_name):
    """
   Compute the membership similarity of each embedding to a specific concept.

   This function calculates how closely each point in the latent space Z aligns
   with the centroid of a given category (type_name). The centroid is the mean
   embedding of all points belonging to that category. Similarity is measured
   using cosine similarity, yielding a score between -1 (opposite) and 1 (identical).

   Args:
       Z (torch.Tensor): Latent embeddings for all examples, shape (num_samples, latent_dim).
       labels (array-like): Array of category labels corresponding to each embedding in Z.
       type_name (str): The category for which membership similarity is computed.

   Returns:
       np.ndarray or None:
           - Array of cosine similarity scores for each embedding relative to the centroid.
           - Returns None if no examples of the specified type are found.
   """
    # Compute cosine similarity of each point to the centroid of a given type
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
    """
    Projects latent embeddings into 2D using PCA and plots them.

    Args:
        Z (torch.Tensor): Latent embeddings of the data (num_samples x latent_dim).
        labeled_mask (array-like or torch.Tensor): Boolean array indicating which samples are labeled.

    Returns:
        None. Displays a scatter plot of the 2D PCA projection.
    """
    # Project latent embeddings to 2D using PCA and plot
    Z_np = Z.detach().numpy()
    Z_2d = PCA(n_components=2, random_state=42).fit_transform(Z_np)
    plt.figure(figsize=(8,6))
    plt.scatter(Z_2d[:,0], Z_2d[:,1], c=labeled_mask, cmap='coolwarm', s=50)
    plt.title("PCA projection of latent flux space (labeled vs unlabeled)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.show()

def visualize_gradients(X_tensor, feature_names):
    """
    Computes the gradient of the latent embeddings with respect to input features
    to estimate feature importance, then visualizes them as a bar plot.

    Args:
        X_tensor (torch.Tensor): Input features with requires_grad enabled.
                                Must have a `.model` attribute pointing to the FluxNet.
        feature_names (list of str): Names of the input features corresponding to columns in X_tensor.

    Returns:
        None. Displays a bar plot of gradient magnitudes for each feature.
    """
    # Compute gradients of latent embeddings w.r.t. input features
    X_tensor.requires_grad_(True)
    Z_grad = X_tensor.model(X_tensor)  # NOTE: assumes model is attached externally
    Z_sum = Z_grad.sum(dim=1)
    Z_sum.backward(torch.ones_like(Z_sum))
    gradients = X_tensor.grad.detach().numpy()
    grad_mag = np.linalg.norm(gradients, axis=0)  # feature importance
    # Plot gradient magnitudes for each feature
    plt.figure(figsize=(10,4))
    plt.bar(range(len(grad_mag)), grad_mag)
    plt.xticks(range(len(grad_mag)), feature_names, rotation=90)
    plt.ylabel("Gradient magnitude")
    plt.title("Feature contributions to latent flux")
    plt.tight_layout()
    plt.show()
