# Deleuzian Neural Network
## Introduction

Modern machine learning systems often rely on Platonic or Aristotelian ontologies, treating concepts as fixed forms or sets of attributes. While effective for traditional classification, such frameworks struggle with nuance, ambiguity, and emergent structures, often encoding human biases and failing to capture the complexity of reality. In contrast, postmodern philosophies, particularly those of Nietzsche and Deleuze, emphasize concepts as dynamic, relational, and defined by their differential interactions rather than fixed identities. This project implements a computational analogue of this ontology, modelling concepts as latent vector fields shaped by differences, enabling more flexible, nuanced, and emergent representations of categories and taxonomy.


## Contents  

1. [Introduction](#introduction)  
2. [Background](#background)  
3. [Installation](#installation)  
4. [Data](#data)  
5. [Model Architecture](#model-architecture)  
6. [Training Objective](#training-objective)  
7. [Emergent Concepts](#emergent-concepts)  
8. [Evaluation](#evaluation)  
9. [Visualization](#visualization)  
10. [Limitations and Future Work](#limitations-and-future-work)  
11. [References](#references)  

## Background

Recent research suggests that deep neural networks, despite differences in architecture and training objectives, converge toward shared representations of reality. These *Platonic representations* resemble Plato’s ideal forms: abstract, invariant, and aligned across modalities such as vision and language. As models scale in size and data diversity, this convergence increases, producing vector embeddings that statistically capture the underlying structure of observed phenomena [1].

Plato’s ontology can be understood by considering philosophical notions of difference. If we have entity A and entity B, they differ with respect to some quality. For example, consider two cats,one black and one white. They differ in color but are both recognized as cats because they share essential qualities such as fur, four legs, and a tail. Aristotle’s law of identity formalizes this: for every category, there are essential qualities uniting all instances.

Essentialism creates a problem when defining a concept by its **qualities** (comprehension) and the **things it applies to** (extension). Take *cat*: its qualities might include eyes, tail, fur, and whiskers, and its extension is all actual cats. The more specific the qualities, the fewer things fit the definition, e.g., requiring fur excludes hairless breeds. Plato addressed this with his **realm of ideal forms**: a perfect, abstract cat embodies all essential qualities, even if no real-world instance exists. This reconciles qualities and examples conceptually but is unobservable.

Another limitation of essentialism is that entities exist in time and undergo qualitative change. A tree, for instance, sprouts leaves in spring, sheds them in autumn, and changes size and shape over decades. Though we still call it “the same tree,” its qualities have evolved, revealing that strict essentialism struggles to account for temporal transformation.

Hegel addresses change through **dialectical or relational difference**, where an entity is defined not by fixed essence but by its relation to what it is not, its negation. A tree is a tree because it is **not a cactus, not a daffodil**, and differs from its past states and environment. This allows coherent logic: the tree can change while remaining recognizable, since identity is relational rather than absolute.

A key limitation of this framework is that identity is still defined in relation to something else. For example, a hybrid plant with traits of both a tree and a cactus challenges classification, because identity depends on pre-existing categories. This reflects a deeper limitation of logic: essences, negation, and fixed categories do not exist in natur, they are human constructs. Taxonomy is a tool for communication, not a mirror of reality. Quantum field theory illustrates this: particles appear stable but are fundamentally unstable distortions of underlying fields. A particle delineates itself from a field without the field delineating itself from the particle; in reality, there are no fixed things, only events [2].

This motivates Nietzsche and Deleuze’s focus on **pure difference**, where identity emerges from difference itself rather than through comparison or negation. Concepts are not defined by fixed qualities or opposition to other categories; they are dynamic patterns of relations and variations. A concept is always in motion, shaped by differences from other phenomena, allowing new identities to emerge organically. For example, a hybrid plant with characteristics of both a tree and a cactus exists as its own concept, without being forced into pre-existing categories, capturing novelty and fluidity naturally [3].

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
```
 


## References 
[1] Huh, M., Cheung, B., Wang, T. & Isola, P., 2024. *The Platonic Representation Hypothesis*. arXiv preprint arXiv:2405.07987.

[2] Lastrevio, 2024. *Quantum Field Theory and Hegel’s Mistakes: How Process Philosophy Helps Solve the Paradoxes of Modern Physics*. Available at: https://lastreviotheory.medium.com/quantum-field-theory-and-hegels-mistakes-how-process-philosophy-helps-solve-the-paradoxes-of-87322def8aa6 [Accessed 24 Sep. 2025].

[3] Deleuze, G., 1983. *Nietzsche and Philosophy*. New York: Columbia University Press.
