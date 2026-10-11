# Silicon Wafer Defect Recognition

## Overview

This project develops a convolutional neural network (CNN) to classify silicon wafer-map images into nine defect classes using the WM-811K dataset.

Automated wafer-defect classification could support semiconductor manufacturing by helping identify defect patterns for further investigation and process improvement.

## Author

**Sabrina Rustgi**  
M.S. Materials Science and Engineering, University of Pennsylvania

## Methods

**Model:** Custom Convolutional Neural Network (CNN)

**Frameworks:** PyTorch, torchvision

### Architecture
- 2D convolutional layers
- Batch normalization
- ReLU activation functions
- Max pooling
- Fully connected classification layer

### Training
- **Loss function:** Cross-entropy loss
- **Optimizer:** AdamW
- **Input image size:** 100 × 100 pixels
- **Output:** Nine defect classes

### Evaluation
- Training and test loss
- Classification accuracy
- Confusion matrix analysis to examine class-specific prediction errors

## Future Work
- Improve model evaluation using additional classification metrics
- Investigate misclassified defect classes and model performance across categories
- Develop a Streamlit application for interactive wafer-map defect classification and visualization
