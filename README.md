# Deleuzian Neural Network
## Introduction

Modern machine learning systems often rely on Platonic or Aristotelian ontologies, treating concepts as fixed forms or sets of attributes. While effective for traditional classification, such frameworks struggle with nuance, ambiguity, and emergent structures, often encoding human biases and failing to capture the complexity of reality. In contrast, postmodern philosophies—particularly those of Nietzsche and Deleuze, emphasize concepts as dynamic, relational, and defined by their differential interactions rather than fixed identities. This project implements a computational analogue of this ontology, modelling concepts as latent vector fields shaped by differences, enabling more flexible, nuanced, and emergent representations of categories and taxonomy.

## Contents 
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

In other words, the data provides the **raw differences**—quantitative and qualitative traits—while the model organizes these differences into dynamic concepts, without presupposing fixed essences.  
 


## References 
[1] Huh, M., Cheung, B., Wang, T. & Isola, P., 2024. *The Platonic Representation Hypothesis*. arXiv preprint arXiv:2405.07987.

[2] Lastrevio, 2024. *Quantum Field Theory and Hegel’s Mistakes: How Process Philosophy Helps Solve the Paradoxes of Modern Physics*. Available at: https://lastreviotheory.medium.com/quantum-field-theory-and-hegels-mistakes-how-process-philosophy-helps-solve-the-paradoxes-of-87322def8aa6 [Accessed 24 Sep. 2025].

[3] Deleuze, G., 1983. *Nietzsche and Philosophy*. New York: Columbia University Press.
