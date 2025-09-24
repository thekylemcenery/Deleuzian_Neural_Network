# =========================================
# main.py
# =========================================

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from functions import *  # assumes functions.py contains your helper functions

# =========================================
# 1. LOAD DATA
# =========================================
df = pd.read_csv("animals_100.tsv", sep="\t")
test_df = pd.read_csv("animals_test.tsv", sep="\t")
numeric_cols = ['Fur', 'Legs', 'Wings', 'Tail', 'Weight']
categorical_cols = ['Color', 'Habitat', 'Diet']

# =========================================
# 2. PREPROCESS FEATURES
# =========================================
X, scaler, encoder = preprocess_features(df, numeric_cols, categorical_cols)
X_tensor = torch.tensor(X, dtype=torch.float32)
labels = df['Type_Label'].fillna('Unlabeled').values
labeled_mask = labels != 'Unlabeled'

# =========================================
# 3. DEFINE FLUX NETWORK
# =========================================
input_dim = X.shape[1]
latent_dim = 16
model = FluxNet(input_dim, latent_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# =========================================
# 4. CONTRASTIVE LOSS & PAIRS
# =========================================
pairs, pair_labels = generate_pairs_with_labels(X_tensor, labels)

# =========================================
# 5. TRAINING LOOP
# =========================================
epochs = 100
margin = 1.0

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for (z1, z2), label_val in zip(pairs, pair_labels):
        optimizer.zero_grad()
        z1_latent = model(z1.unsqueeze(0))
        z2_latent = model(z2.unsqueeze(0))
        loss = contrastive_loss(z1_latent, z2_latent, label_val.unsqueeze(0), margin)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        avg_loss = total_loss / max(1, len(pairs))
        print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")

# =========================================
# 6. EMERGENT CONCEPTS
# =========================================
model.eval()
with torch.no_grad():
    Z = model(X_tensor)

# compute centroids for each type
type_names = ['Cat','Dog','Bird','Aquatic Mammal']
centroids = {}
for t in type_names:
    idx = [i for i,l in enumerate(labels) if l==t]
    if idx:
        centroids[t] = Z[idx].mean(dim=0)

# =========================================
# 7. PREDICT TEST ANIMALS
# =========================================
X_test, _, _ = preprocess_features(test_df, numeric_cols, categorical_cols, scaler=scaler, encoder=encoder)
with torch.no_grad():
    Z_test = model(torch.tensor(X_test, dtype=torch.float32))

# kNN on latent embeddings
train_idx = np.where(labeled_mask)[0]
train_embeddings = Z[train_idx].detach().numpy()
train_labels = labels[train_idx]
pca_knn = PCA(n_components=min(8, latent_dim))
train_embeddings_pca = pca_knn.fit_transform(train_embeddings)
test_embeddings_pca = pca_knn.transform(Z_test.detach().numpy())
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_embeddings_pca, train_labels)
predicted_labels = knn.predict(test_embeddings_pca)
test_df["Predicted_Label"] = predicted_labels
print("\nPredicted categories for test animals:")
print(test_df[["Animal","Predicted_Label"]])

# =========================================
# 8. EVALUATE IF LABELED TEST DATA EXISTS
# =========================================
try:
    labeled_test_df = pd.read_csv("animals_test_labeled.tsv", sep="\t")
    true_labels = labeled_test_df["Type_Label"].values
    pred_eval = test_df["Predicted_Label"].values[:len(true_labels)]
    pct_false = np.mean(pred_eval != true_labels)*100
    print(f"\nPercentage of false labels: {pct_false:.2f}%")
except FileNotFoundError:
    print("\nNo labeled test file found. Skipping evaluation.")

# =========================================
# 9. VISUALIZE CONCEPT VECTOR FIELDS
# =========================================
# PCA to 2D for visualization
pca_2d = PCA(n_components=2, random_state=42)
Z_2d = pca_2d.fit_transform(Z.detach().numpy())
centroids_2d = {k: pca_2d.transform(v.unsqueeze(0).numpy())[0] for k,v in centroids.items()}

# grid for vector field
x_min, x_max = Z_2d[:,0].min()-1, Z_2d[:,0].max()+1
y_min, y_max = Z_2d[:,1].min()-1, Z_2d[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min,x_max,20), np.linspace(y_min,y_max,20))
U = np.zeros_like(xx)
V = np.zeros_like(yy)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        point = np.array([xx[i,j], yy[i,j]])
        closest_c = min(centroids_2d, key=lambda k: np.linalg.norm(centroids_2d[k]-point))
        vec = centroids_2d[closest_c] - point
        U[i,j] = vec[0]
        V[i,j] = vec[1]

plt.figure(figsize=(8,6))
plt.quiver(xx, yy, U, V, color='teal', alpha=0.6)
plt.scatter(Z_2d[:,0], Z_2d[:,1], c='red', label='Training animals')
for name,pos in centroids_2d.items():
    plt.scatter(pos[0], pos[1], label=name, s=100)
plt.title("Latent Concept Vector Field (Deleuzian Flux)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend()
plt.show()
