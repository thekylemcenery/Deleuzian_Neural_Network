# Deleuzian Neural Network
## Introduction

Modern machine learning systems often rely on Platonic or Aristotelian ontologies, treating concepts as fixed forms or sets of attributes. While effective for traditional classification, such frameworks struggle with nuance, ambiguity, and emergent structures, often encoding human biases and failing to capture the complexity of reality. In contrast, postmodern philosophies, particularly those of Nietzsche and Deleuze, emphasize concepts as dynamic, relational, and defined by their differential interactions rather than fixed identities. This project implements a computational analogue of this ontology, modelling concepts as latent vector fields shaped by differences, enabling more flexible, nuanced, and emergent representations of categories and taxonomy.


## Contents

1. [Introduction](#introduction)
2. [Background](#background)
3. [Installation](#installation)
4. [Data](#data)
5. [Model Architecture](#model-architecture)
6. [Training the FluxNet](#training-the-fluxnet)
7. [Emergent Concepts](#emergent-concepts)
8. [Evaluation](#evaluation)
9. [Visualisation](#visualisation)
10. [Outlook](#outlook)
11. [Acknowledgements](#acknowledgements)
12. [References](#references)

## Background

Recent research suggests that deep neural networks, despite differences in architecture and training objectives, converge toward shared representations of reality. These *Platonic representations* resemble Plato’s ideal forms: abstract, invariant, and aligned across modalities such as vision and language. As models scale in size and data diversity, this convergence increases, producing vector embeddings that statistically capture the underlying structure of observed phenomena [1].

Plato’s ontology can be understood by considering philosophical notions of difference. If we have entity A and entity B, they differ with respect to some quality. For example, consider two cats,one black and one white. They differ in color but are both recognized as cats because they share essential qualities such as fur, four legs, and a tail. Aristotle’s law of identity formalizes this: for every category, there are essential qualities uniting all instances.

Essentialism creates a problem when defining a concept by its **qualities** (comprehension) and the **things it applies to** (extension). Take *cat*: its qualities might include eyes, tail, fur, and whiskers, and its extension is all actual cats. The more specific the qualities, the fewer things fit the definition, e.g., requiring fur excludes hairless breeds. Plato addressed this with his **realm of ideal forms**: a perfect, abstract cat embodies all essential qualities, even if no real-world instance exists. This reconciles qualities and examples conceptually but is unobservable.

Another limitation of essentialism is that entities exist in time and undergo qualitative change. A tree, for instance, sprouts leaves in spring, sheds them in autumn, and changes size and shape over decades. Though we still call it “the same tree,” its qualities have evolved, revealing that strict essentialism struggles to account for temporal transformation.

Hegel addresses change through **dialectical or relational difference**, where an entity is defined not by fixed essence but by its relation to what it is not, its negation. A tree is a tree because it is **not a cactus, not a daffodil**, and differs from its past states and environment. This allows coherent logic: the tree can change while remaining recognizable, since identity is relational rather than absolute.

A key limitation of this framework is that identity is still defined in relation to something else. For example, a hybrid plant with traits of both a tree and a cactus challenges classification, because identity depends on pre-existing categories. This reflects a deeper limitation of logic: essences, negation, and fixed categories do not exist in natur, they are human constructs. Taxonomy is a tool for communication, not a mirror of reality. Quantum field theory illustrates this: particles appear stable but are fundamentally unstable distortions of underlying fields. A particle delineates itself from a field without the field delineating itself from the particle; in reality, there are no fixed things, only events [2].

This motivates Nietzsche and Deleuze’s focus on **pure difference**, where identity emerges from difference itself rather than through comparison or negation. Concepts are not defined by fixed qualities or opposition to other categories; they are dynamic patterns of relations and variations. A concept is always in motion, shaped by differences from other phenomena, allowing new identities to emerge organically. For example, a hybrid plant with characteristics of both a tree and a cactus exists as its own concept, without being forced into pre-existing categories, capturing novelty and fluidity naturally [3]. 

Deleuze fully realised this notion of "difference in itself", where difference is metaphysically and ontologically prior to identity in his 1968 work "Difference and Repetition".[4] He later dubbed this ontology as ‘transcendental empiricism’ in contrast to Kant's notion of 'transcendental idealism. This project is a first attempt to explore these ideas mathematically, following Deleuze’s analogy with Leibniz, difefrnce is best understood via dx, the differential. The derivative, dy/dx, determines the structure of a curve while existing outside the curve itself; that is, by describing a virtual tangent. Analogously, if a concept's **comprehension** can be understood in terms of its features, these features can be represented as **variables**. However, for the **entire extension** of the concept to be captured in its general definition, the definition itself must be **dynamic with respect to context**. Thus, we define a concept as a **dynamic vector field**, such that **all examples within the extension** (represented by vectors) can be **understood within the definition**, without imposing the **closed boundaries** associated with traditional essentialism.

## Installation

To run this project, you need Python 3.8+ and the following libraries. You can install them using `pip`.

```bash
# Core libraries
pip install numpy pandas matplotlib

# PyTorch (replace with your CUDA version if needed)
pip install torch torchvision torchaudio

# Scikit-learn for preprocessing, PCA, and kNN
pip install scikit-learn
```


## Data

The project uses two tab-separated datasets:  

- **`animals_100.tsv`**: the main training dataset, containing around 100 animals.  
- **`animals_test.tsv`**: an unlabeled test set for evaluating emergent category assignments.  
- *(Optional)* **`animals_test_labeled.tsv`**: a labeled version of the test set, if available, used to compute prediction accuracy.  

Each animal is described through two kinds of features:  

- **Numeric features**: `Fur`, `Legs`, `Wings`, `Tail`, `Weight` — measurable traits.  
- **Categorical features**: `Color`, `Habitat`, `Diet` — contextual traits.  

Labels (e.g. *Cat*, *Dog*, *Bird*, *Aquatic Mammal*) are provided for some animals but not all. This partial labeling is intentional: the network is not meant to learn fixed categories in the Aristotelian sense, but to generate **emergent clusters** in latent space.  

In other words, the data provides the **raw differences** (quantitative and qualitative traits), while the model organizes these differences into dynamic concepts, without presupposing fixed essences.  


## Model Architecture  

At the heart of this project is **FluxNet**, a simple feedforward neural network that compresses raw input features into a lower-dimensional *latent flux space*. Unlike classical models that fix categories in advance, this network learns relational embeddings that allow categories to emerge dynamically.

```python
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
```

Contrastive loss trains the network to organize embeddings based on relational similarity, reflecting the idea that identity emerges from difference. For each pair of embeddings `z1` and `z2`, we compute their cosine similarity (`cos_sim`), which ranges from -1 (opposite) to 1 (identical). Positive pairs (`label=1`) are encouraged to be close in latent space using a loss of `1 - cos_sim`, while negative pairs (`label=0`) are pushed apart only if their similarity exceeds a margin (`torch.clamp(cos_sim - margin, min=0)`). The mean of these positive and negative losses is taken over all pairs to train the network. Intuitively, cats cluster near other cats, and cats are pushed away from birds or dogs. This allows the latent space to emerge as a Deleuzian flux, where concepts form dynamically according to differences rather than fixed definitions.

```python
def contrastive_loss(z1, z2, label, margin=1.0):
    # Compute cosine similarity between two embeddings
    cos_sim = nn.functional.cosine_similarity(z1, z2)
    # If same label: minimize (1 - similarity)
    pos_loss = (1 - cos_sim) * label
    # If different label: penalize similarity beyond margin
    neg_loss = torch.clamp(cos_sim - margin, min=0) * (1 - label)
    # Average the combined loss
    return (pos_loss + neg_loss).mean()
```

The `generate_pairs_with_labels` function prepares the input for contrastive loss by creating all unique pairs of examples from the dataset. For each pair, it assigns a label: `1` if both examples belong to the same category (positive pair) and `0` if they belong to different categories (negative pair). The function iterates through every combination of examples, skipping any unlabeled entries, and returns two lists: `pair_list`, containing the pairs of feature vectors, and `label_list`, containing the corresponding 0/1 labels as a tensor. This ensures that the network receives both similar and dissimilar examples to learn a latent space where embeddings reflect conceptual differences and similarities dynamically.

```python
def generate_pairs_with_labels(X_tensor, labels):
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
```

The `preprocess_features` function converts a raw DataFrame into a clean, numeric feature matrix suitable for the FluxNet neural network. It handles both **numeric** and **categorical** columns:

1. **Numeric preprocessing**  
   - Forces numeric columns to numeric types (`pd.to_numeric`) and coerces errors.  
   - Fills missing values with the mean of each column.  
   - Standardizes numeric features (zero mean, unit variance) using `StandardScaler`. If a scaler is provided, it reuses it; otherwise, it fits a new one.

2. **Categorical preprocessing**  
   - Converts categorical columns into one-hot encoded vectors using `OneHotEncoder`.  
   - Handles unknown categories gracefully if using a previously fitted encoder.

3. **Feature combination**  
   - Concatenates numeric and categorical features into a single matrix `X` for input to the neural network.

The function returns the feature matrix `X` along with the fitted `scaler` and `encoder` for consistent preprocessing of test data.

```python
def preprocess_features(df, numeric_cols, categorical_cols, scaler=None, encoder=None):
    # --- Numeric preprocessing ---
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')   
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())  
    if not scaler:
        scaler = StandardScaler()                          
        num_features = scaler.fit_transform(df[numeric_cols])
    else:
        num_features = scaler.transform(df[numeric_cols])  

    # --- Categorical preprocessing ---
    if not encoder:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        context_features = encoder.fit_transform(df[categorical_cols])
    else:
        context_features = encoder.transform(df[categorical_cols])

    # Combine numeric + categorical into final feature matrix
    X = np.hstack([num_features, context_features])
    return X, scaler, encoder
```

The `compute_membership` function calculates the degree to which each embedding in the latent space belongs to a specific category or concept. It does this by first identifying all embeddings with the target label (`type_name`) and computing their **centroid**—the average position in the latent space. Then, it measures the **cosine similarity** between each point in the full embedding set and this centroid. Cosine similarity ranges from -1 to 1, where 1 indicates that a point is perfectly aligned with the concept centroid, 0 indicates no alignment, and -1 indicates opposition. 

Intuitively, this allows us to see which points in the latent space “fit” a concept, even for ambiguous or partially labeled data. For example, if we compute membership for the concept *Cat*, embeddings representing black cats, white cats, and other feline variations will have higher similarity scores, while embeddings for dogs or birds will have lower scores. This forms the basis of the **concept vector field**, which visualizes categories dynamically according to the differences between instances, reflecting a Deleuzian notion of flux and emergent identity.

```python
def compute_membership(Z, labels, type_name):
    # Compute cosine similarity of each point to the centroid of a given type
    idx = [i for i, l in enumerate(labels) if l == type_name]
    if not idx:
        return None
    centroid = Z[idx].mean(dim=0)
    sim = nn.functional.cosine_similarity(Z, centroid.unsqueeze(0))
    return sim.numpy()

```
## Training the FluxNet

The first step is to load both the training and test datasets. In this project, the training data (`animals_100.tsv`) contains animal features along with their category labels, while the test data (`animals_test.tsv`) contains the same features but may be partially or completely unlabeled.  

Features are split into two types:

- **Numeric features**: measurable quantities such as `'Fur'`, `'Legs'`, `'Wings'`, `'Tail'`, and `'Weight'`.  
- **Categorical features**: descriptive properties such as `'Color'`, `'Habitat'`, and `'Diet'`.

This separation is important because numeric and categorical features require different preprocessing steps for input into a neural network.

```python
df = pd.read_csv("animals_100.tsv", sep="\t")     # training dataset
test_df = pd.read_csv("animals_test.tsv", sep="\t")  # test dataset
numeric_cols = ['Fur', 'Legs', 'Wings', 'Tail', 'Weight']   # numeric features
categorical_cols = ['Color', 'Habitat', 'Diet']             # categorical features
```

Once the raw data is loaded, the features must be prepared for input into the neural network. This section performs **numeric standardization**, **categorical one-hot encoding**, and converts the data into a PyTorch tensor.

```python
# Standardize numeric features + one-hot encode categorical features
X, scaler, encoder = preprocess_features(df, numeric_cols, categorical_cols)
X_tensor = torch.tensor(X, dtype=torch.float32)
labels = df['Type_Label'].fillna('Unlabeled').values   # labels (use "Unlabeled" if missing)
labeled_mask = labels != 'Unlabeled'                  # boolean mask: labeled vs unlabeled
```
After preprocessing, we need to set up the neural network that will learn **latent concept representations** from the input features.

```python
# Input dimension = number of processed features
input_dim = X.shape[1]
latent_dim = 16   # dimension of latent flux space
model = FluxNet(input_dim, latent_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

Contrastive learning requires pairs of examples with a label indicating whether they belong to the same category (positive pair) or different categories (negative pair).

```python
# Create training pairs (same-label = positive, diff-label = negative)
pairs, pair_labels = generate_pairs_with_labels(X_tensor, labels)
```

The following section trains the FluxNet model using **contrastive learning**, where the network learns a latent space that encodes conceptual similarity and difference. Each pair of examples is used to teach the model which samples should be close together (same category) and which should be farther apart (different categories). Over multiple epochs, the network gradually adjusts its weights to form a meaningful **latent flux space** in which emergent concepts are geometrically represented.

```python
epochs = 100
margin = 1.0   # contrastive loss margin

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for (z1, z2), label_val in zip(pairs, pair_labels):
        optimizer.zero_grad()
        # encode both items into latent space
        z1_latent = model(z1.unsqueeze(0))
        z2_latent = model(z2.unsqueeze(0))
        # compute contrastive loss
        loss = contrastive_loss(z1_latent, z2_latent, label_val.unsqueeze(0), margin)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # print average loss every 10 epochs
    if epoch % 10 == 0:
        avg_loss = total_loss / max(1, len(pairs))
        print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")
```

## Emergent Concepts
After training, the network is used to compute **latent embeddings** for all training examples. Setting `model.eval()` and wrapping the computation in `torch.no_grad()` ensures that the network runs in inference mode without tracking gradients.  

```python
model.eval()
with torch.no_grad():
    Z = model(X_tensor)  # latent embeddings of training data

# Compute centroids (concept centers) for each known type
type_names = ['Cat','Dog','Bird','Aquatic Mammal']
centroids = {}
for t in type_names:
    idx = [i for i,l in enumerate(labels) if l==t]
    if idx:
        centroids[t] = Z[idx].mean(dim=0)
```
## Evaluation
### 7. Predicting Test Animals

Once the network has learned latent concept representations, it can be used to classify new, unseen data. First, the test set is preprocessed using the **same scaler and encoder** fitted on the training data, ensuring consistency in numeric scaling and one-hot encoding of categorical features. The network then maps the test examples into the **latent flux space** without computing gradients:

```python
X_test, _, _ = preprocess_features(test_df, numeric_cols, categorical_cols, scaler=scaler, encoder=encoder)
with torch.no_grad():
    Z_test = model(torch.tensor(X_test, dtype=torch.float32))

# Train a kNN classifier on latent embeddings of labeled data
train_idx = np.where(labeled_mask)[0]
train_embeddings = Z[train_idx].detach().numpy()
train_labels = labels[train_idx]

# Use PCA to reduce dimensionality before kNN (helps stability)
pca_knn = PCA(n_components=min(8, latent_dim))
train_embeddings_pca = pca_knn.fit_transform(train_embeddings)
test_embeddings_pca = pca_knn.transform(Z_test.detach().numpy())

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_embeddings_pca, train_labels)
predicted_labels = knn.predict(test_embeddings_pca)
test_df["Predicted_Label"] = predicted_labels
print("\nPredicted categories for test animals:")
print(test_df[["Animal","Predicted_Label"]])
```


If a labeled test set is available, the model’s predictions can be quantitatively evaluated. The code first attempts to load the labeled test file. If found, it compares the predicted labels from the kNN classifier to the true labels:

```python
try:
    labeled_test_df = pd.read_csv("animals_test_labeled.tsv", sep="\t")
    true_labels = labeled_test_df["Type_Label"].values
    pred_eval = test_df["Predicted_Label"].values[:len(true_labels)]
    pct_false = np.mean(pred_eval != true_labels)*100
    print(f"\nPercentage of false labels: {pct_false:.2f}%")
except FileNotFoundError:
    print("\nNo labeled test file found. Skipping evaluation.")
```

## Visualisation


The final step visualizes the learned **latent flux space** and how test examples relate to emergent concepts. First, the latent embeddings of the training data are projected into 2D using PCA for easier visualization:

```python
# PCA to 2D for plotting latent embeddings
pca_2d = PCA(n_components=2, random_state=42)
Z_2d = pca_2d.fit_transform(Z.detach().numpy())
centroids_2d = {k: pca_2d.transform(v.unsqueeze(0).numpy())[0] for k,v in centroids.items()}
```
Next, a vector field is generated across a grid of points in the 2D space. Each arrow points toward the nearest concept centroid, illustrating how the latent space “flows” toward emergent categories:

```python
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
```
Finally, the vector field is plotted along with the training points and centroids:

```python
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
```

This resulting visualization shows how the latent embeddings organize into concept “attractors,” with arrows indicating the flow toward centroids. Outliers in the test data, such as a hairless cat or an unusually heavy bird—appear away from the main clusters, highlighting how extreme or novel examples are naturally represented without forcing them into existing categories. 

<img width="506" height="387" alt="Figure 1" src="https://github.com/user-attachments/assets/80e905d8-cbf5-4c40-a390-00c5ad050815" />
 
## Outlook

The model’s predictions showed mixed success, which is unsurprising given the exploratory nature of this project. The primary goal was not to optimize classification accuracy, but rather to investigate how Deleuze’s ontology could be represented mathematically in a neural network. A full-scale implementation with robust error minimization would be needed to improve predictive performance, but that was beyond the scope of this study. Think of this work as a conceptual exploration of a novel approach to **information ontology**. For future development, one could design a more focused neural network that learns a single concept at a time using a semi-supervised dataset from a single perceptual category. This would allow the model to produce a **single, coherent vector field** representing that concept, offering a cleaner, more interpretable latent space and a more faithful implementation of Deleuzian flux.


## Acknowledgments

Thanks to [Vrushal](https://github.com/Vrushall) for contributions and feedback on this project.


## References 
[1] Huh, M., Cheung, B., Wang, T. & Isola, P., 2024. *The Platonic Representation Hypothesis*. arXiv preprint arXiv:2405.07987.

[2] Lastrevio, 2024. *Quantum Field Theory and Hegel’s Mistakes: How Process Philosophy Helps Solve the Paradoxes of Modern Physics*. Available at: https://lastreviotheory.medium.com/quantum-field-theory-and-hegels-mistakes-how-process-philosophy-helps-solve-the-paradoxes-of-87322def8aa6 [Accessed 24 Sep. 2025].

[3] Deleuze, G., 1983. *Nietzsche and Philosophy*. New York: Columbia University Press.

[4] Deleuze, G. (1968). Difference and Repetition. Paris: Presses Universitaires de France.
