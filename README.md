# Image Classification Using Latent from VAE

This project implements a two-stage approach for image classification using Variational Autoencoders (VAE) and a Multi-Layer Perceptron (MLP). First, a VAE learns a compressed latent representation of the images, then these latent vectors are used to train an MLP classifier.

## Architecture Overview

### Stage 1: Variational Autoencoder (VAE)
- **Encoder**: Convolutional neural network that maps images to latent space
  - 5 convolutional layers with increasing filters (32→512)
  - BatchNorm and LeakyReLU activation
  - Outputs mean (μ) and log variance (logσ) of latent distribution

- **Decoder**: Deconvolutional network that reconstructs images from latent vectors
  - Linear layer to reshape latent vector
  - 5 upsampling + conv layers with decreasing filters (512→3)
  - Instance normalization and LeakyReLU activation
  - Tanh output activation

### Stage 2: MLP Classifier
- Takes VAE latent vectors (128-dim) as input
- 3 fully connected layers (128→128→75)
- ReLU activation and dropout for regularization
- Outputs class probabilities

## Features

- Supports both butterfly and animal image datasets
- Multiple latent samples per input for robust representation
- Configurable hyperparameters (batch size, learning rate, etc.)
- TensorBoard integration for loss visualization
- Gradient tracking and model checkpointing
- Evaluation metrics: Accuracy and F1-Score

## Requirements

```bash
torch
torchvision
torchmetrics
tensorboard
PIL
numpy
matplotlib
tqdm
```

## Usage

1. Set up the environment:
```bash
pip install -r requirements.txt
```

2. Configure hyperparameters in the script:
```python
IMG_SIZE = 128
LATENT_DIM = 128
NUM_EPOCHS = 250
BATCH_SIZE = 128
```

3. Train the VAE:
```python
TRAIN_VAE = True
TRAIN_MLP = False
```

4. Train the MLP classifier:
```python
TRAIN_VAE = False
TRAIN_MLP = True
```

## Model Training

The training process is split into two phases:

1. **VAE Training**:
   - Optimizes reconstruction loss and KL divergence
   - Generates sample images during training
   - Saves model checkpoints

2. **MLP Training**:
   - Uses frozen VAE encoder to generate latent vectors
   - Trains classifier on these latent representations
   - Monitors accuracy and F1-score

## Results Visualization

- Generated and reconstructed images saved during training
- TensorBoard logs for loss components
- Training/validation metrics plotting

## Directory Structure

```
.
├── data/
│   ├── Animals_data/
│   └── butterfly_data/
├── VAE/
│   ├── models/
│   ├── generated_images/
│   └── tensorboard/
└── notebooks/
    └── Image classification using latent from VAE.ipynb
```

## License

[MIT License](LICENSE)
