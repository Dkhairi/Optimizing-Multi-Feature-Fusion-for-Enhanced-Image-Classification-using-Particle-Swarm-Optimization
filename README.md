# Optimizing-Multi-Feature-Fusion-for-Enhanced-Image-Classification-using-Particle-Swarm-Optimization

This repository contains code for a multi-feature fusion approach for image classification, leveraging Particle Swarm Optimization (PSO) to determine optimal feature weights. The methodology is designed to enhance classification accuracy by combining diverse feature sets extracted from images.

Overview
This project implements a comprehensive image classification pipeline that includes:

Data Pre-processing: Conversion of RGB images to grayscale and normalization.
Feature Extraction:
Gray-Level Co-occurrence Matrices (GLCM) for texture features.
Discrete Wavelet Transform (DWT) and Stationary Wavelet Transform (SWT) for multi-scale features.
Convolutional Neural Network (CNN) for deep feature extraction using VGG16.
PSO-based Weight Optimization: Utilizing PSO to find the optimal weights for the extracted feature sets.
Classification: Training and evaluating classifiers (SVM, Random Forest, Shallow CNN) on the fused feature vectors.
The methodology is evaluated on two datasets:

Kaggle Fruits Dataset: For fruit quality classification. https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification
KTH-TIPS2-b Dataset: For industrial materials image classification. https://service.tib.eu/ldmservice/dataset/kth-tips2-b
Methodology
Data Pre-Processing
Images are converted to grayscale and normalized to ensure consistency and reduce computational complexity.

Feature Extraction
GLCM-based Texture Features: Extracts texture features such as Contrast, Correlation, Energy, and Homogeneity.
Wavelet Transformation-based Features: Applies DWT and SWT to extract multi-scale features, including statistical measures from sub-bands.
DCNN-based Feature Extraction: Utilizes a pre-trained VGG16 model to extract deep features.
PSO-based Weight Optimization
PSO is used to optimize the weights of the extracted feature sets, enhancing the classification performance by assigning higher weights to more informative features.

Classifiers Training and Evaluation
Classifiers such as SVM, Random Forest, and Shallow CNN are trained and evaluated on the optimized feature vectors.

Datasets
Kaggle Fruits Dataset: Comprising images of fresh and rotten fruits.
KTH-TIPS2-b Dataset: Containing images of various industrial materials.
Results
The results demonstrate that PSO-based weight optimization significantly improves classification accuracy across all classifiers and datasets.

Significant accuracy improvements are observed when using the PSO-optimized feature vector compared to individual feature sets.
The effectiveness of the proposed approach is validated on two diverse datasets.
Usage
To use this code:

Clone the repository.
Install the required dependencies.
Prepare the datasets and update the file paths in the code.
Run the feature extraction and classification scripts.
