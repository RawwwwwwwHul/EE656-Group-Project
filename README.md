ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection
================================================================================

EE656 Course Project

--------------------------------------------------------------------------------
1. OVERVIEW
--------------------------------------------------------------------------------
This repository contains our implementation and reproduction of ISTVT
(Interpretable Spatial-Temporal Video Transformer), a transformer-based
framework for deepfake video detection. The model combines an Xception-based
feature extractor with a spatial-temporal transformer and a self-subtract
attention mechanism to jointly capture spatial artifacts and temporal
inconsistencies in facial video clips.

We reproduce the core methodology of the original ISTVT paper on a subset of
the FaceForensics++ (C23) dataset, adapting the architecture to run under
consumer-grade GPU constraints (RTX 2050 / Apple M3), and analyze the model's
predictions using Gradient Attention Rollout for interpretability.

We selected this paper because it combines an effective video transformer
architecture with an interpretable attention mechanism, making it both
technically challenging and practically relevant.


--------------------------------------------------------------------------------
2. DATASET
--------------------------------------------------------------------------------
Source: FaceForensics++ (FF++) C23 dataset (obtained via Kaggle)

Manipulation techniques used:
  - DeepFakeDetection
  - Face2Face
  - NeuralTextures
  - FaceSwap
  - None (Real / Original)

Initial composition:
  DeepFakeDetection   : 60 videos
  Face2Face           : 60 videos
  NeuralTextures      : 60 videos
  FaceSwap            : 60 videos
  Real (Original)     : 100 videos

Final composition after preprocessing:
  Real (Original)                : 4500 .pt files
  Fake (all manipulation types)  : 6000 .pt files

Train / Validation / Test split: 70% / 15% / 15% (strict video-level split
to prevent identity/data leakage between sets).


--------------------------------------------------------------------------------
3. PREPROCESSING PIPELINE
--------------------------------------------------------------------------------
Scripts (run in order):

  1. download_ff_subset.py
       Downloads the required subset of the FaceForensics++ C23 dataset.

  2. preprocess.py
       - Performs face detection/extraction using MTCNN
       - Resizes detected facial regions to 128 x 128 pixels
       - Uniformly samples 6 representative frames per video
       - Saves frames + labels + metadata as PyTorch (.pt) tensors

  3. filter_dataset.py
       Balances the dataset by reducing the number of manipulated clips
       while retaining all original (real) clips, producing the final
       balanced dataset of 4500 real / 6000 fake clips.


--------------------------------------------------------------------------------
4. MODEL ARCHITECTURE
--------------------------------------------------------------------------------
Code location: architecture/

  architecture/backbone.py
    - Xception network, Entry Flow only, used as frozen feature extractor
      (requires_grad=False)
    - Input:  6 x 3 x 128 x 128  (6 frames per clip)
    - Output: 6 x 728 x 8 x 8

  architecture/tokenizer.py
    - Converts each 8x8 feature map into 64 flattened spatial patches,
      linearly projected into a 512-dim embedding space
    - Appends a learnable CLS token per frame (64 -> 65 tokens/frame)
    - Adds learnable spatial and temporal positional embeddings
    - Output: 6 x 65 x 512 transformer tokens

  architecture/transformer.py
    - Position-wise Feed-Forward Network (MLP):
        hidden dim = 4x embedding dim (2048 for 512-dim embeddings)
        GELU activation, dropout after each linear layer
    - ISTVTBlock (Pre-LN transformer block), applied in order per block:
        a. Temporal Attention w/ residual
             - self-subtract mechanism applied to Q/K projections
             - multi-head attention across the temporal dimension
        b. Spatial Attention w/ residual
             - standard multi-head self-attention across 65 spatial
               tokens, independently per frame
        c. Feed-Forward Network w/ residual
    - Full model: 6 stacked ISTVTBlock modules

  Classification head:
    - CLS tokens from all 6 frames are concatenated and passed through
      an MLP for binary classification (real vs. fake).
      (Note: the original paper does not specify the final CLS
      aggregation strategy; concatenation was our design choice.)


--------------------------------------------------------------------------------
5. IMPLEMENTATION MODIFICATIONS (vs. original paper)
--------------------------------------------------------------------------------
Due to hardware constraints, the following changes were made relative to the
original ISTVT paper:

  Parameter               Original Paper         This Implementation
  ----------------------  ---------------------  ---------------------
  Input resolution         300 x 300               128 x 128
  Embedding dimension       768                     512
  Transformer blocks        12                      6
  Training epochs           100                     30
  Hardware                  4x Tesla V100           RTX 2050 / Apple M3


--------------------------------------------------------------------------------
6. TRAINING PROTOCOL
--------------------------------------------------------------------------------
  Loss function     : Weighted CrossEntropyLoss
                       (real class weight ~1.33, fake class weight 1.0,
                       to correct for class imbalance: 6000 fake / 4500 real)
  Optimizer         : AdamW
  Learning rate     : 1e-4
  Weight decay      : 1e-4
  Batch size        : 8
  Epochs            : 30
  Dropout           : 0.1
  MLP ratio         : 4 (hidden dim = 2048)
  Embedding dim     : 512
  Attention heads   : 8
  Transformer blocks: 6
  Random seed       : 42 (Python, NumPy, PyTorch, CUDA)
  Data split        : 70% train / 15% val / 15% test (video-level, leak-free)


--------------------------------------------------------------------------------
7. INTERPRETABILITY
--------------------------------------------------------------------------------
Interpretability is implemented via Gradient Attention Rollout (in place of
the original paper's Layer-wise Relevance Propagation / Deep Taylor
Decomposition, which requires modifying the backward pass of every layer).

  - Spatial attention visualization:
      Gradient-weighted attention maps from each spatial transformer block
      are recursively aggregated (rollout) and overlaid as heatmaps on each
      of the 6 input frames, resized to 128 x 128.

  - Temporal attention visualization:
      Gradient-weighted temporal attention matrices are extracted per block
      and aggregated to show how much each of the 6 frames contributes to
      the final prediction.


--------------------------------------------------------------------------------
8. RESULTS
--------------------------------------------------------------------------------
Evaluated on the held-out FF++ C23 test set (1575 samples):

  Metric       Value
  -----------  -------
  Accuracy     98.10%
  Precision    97.49%
  Recall       99.22%
  F1-Score     98.35%

Confusion Matrix:

                    Predicted Real   Predicted Fake
  Actual Real            652              23
  Actual Fake              7             893

The model correctly classified 652/675 real videos and 893/900 manipulated
videos, misclassifying only 23 real and 7 fake videos.


--------------------------------------------------------------------------------
9. REPOSITORY STRUCTURE (suggested)
--------------------------------------------------------------------------------
.
├── download_ff_subset.py        # Downloads FF++ C23 subset
├── preprocess.py                 # Face detection, cropping, frame sampling
├── filter_dataset.py             # Class balancing
├── architecture/
│   ├── backbone.py               # Xception Entry Flow feature extractor
│   ├── tokenizer.py              # Patch embedding + CLS + positional encodings
│   └── transformer.py            # ISTVTBlock, MLP, full transformer stack
├── train.py                       # Training loop / optimization
├── evaluate.py                    # Test-set evaluation & metrics
├── interpretability/
│   └── gradient_rollout.py       # Gradient Attention Rollout (spatial + temporal)
├── figures/                       # Generated attention overlays, confusion matrix
├── report/                        # Project report (PDF)
└── README.txt                     # This file


--------------------------------------------------------------------------------
10. LIMITATIONS
--------------------------------------------------------------------------------
  - Reduced 128x128 input resolution may miss fine-grained artifacts,
    especially for subtle manipulations like NeuralTextures.
  - The Xception backbone was fully frozen; it does not adapt beyond its
    original ImageNet-pretrained features.
  - Evaluation was performed only on FF++ C23; no cross-dataset testing
    was done on Celeb-DF or DFDC.
  - No baseline comparison (e.g., frame-only Xception, C3D) was performed,
    so the isolated contribution of temporal attention is not quantified.
  - Trained for only 30 epochs with batch size 8, versus the original
    paper's 100 epochs on 4x Tesla V100 GPUs.


--------------------------------------------------------------------------------
11. FUTURE WORK
--------------------------------------------------------------------------------
  - Cross-dataset evaluation on Celeb-DF and DFDC
  - Training at higher input resolution
  - Unfreezing the Xception backbone after initial convergence
  - Baseline comparisons against frame-only Xception / C3D models


--------------------------------------------------------------------------------
12. REFERENCE
--------------------------------------------------------------------------------
ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake
Detection (original paper).

--------------------------------------------------------------------------------
