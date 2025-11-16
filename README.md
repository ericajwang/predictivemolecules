A deep learning framework for predicting small molecule-protein binding affinities. This project implements several different neural network architectures (GNNs, Transformers, CNNs) and lets you experiment with ensemble methods to combine their predictions. 
The codebase handles everything from molecular graph construction and fingerprint generation to model training and evaluation.

The framework is designed to be flexible—you can easily switch between graph-based models that leverage molecular structure, transformer models that use attention mechanisms, or CNN models that process molecular fingerprints. 
Everything is configured through YAML files, and the training pipeline includes early stopping, checkpointing, and comprehensive metrics tracking. 
There's also built-in support for ensemble methods if you want to combine multiple models for better predictions.
