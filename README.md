
# Automated Skin Lesion Classification System

## Overview
This project focuses on developing an **Automated Skin Lesion Classification System** that leverages state-of-the-art deep learning models to analyze dermoscopic images of skin lesions. The system is designed to classify lesions into eight distinct classes, providing a reliable and efficient tool for early detection of skin cancer.

### Problem Statement
The increasing prevalence of skin cancer necessitates early and accurate detection to improve survival rates. Manual diagnosis can be error-prone and time-consuming, especially for dermatologists dealing with large volumes of patients. This project aims to develop an automated solution using deep learning models to assist in dermatological decision-making. By leveraging parallelism and distributed computing, the system is optimized for scalability and computational efficiency.

### Dataset
The dataset used in this project is sourced from the [ISIC Archive](https://challenge.isic-archive.com/landing/2019/), a repository of high-resolution dermoscopic images. The dataset consists of thousands of labeled images across eight classes:
1. **Melanoma**
2. **Melanocytic Nevus**
3. **Basal Cell Carcinoma**
4. **Actinic Keratosis**
5. **Benign Keratosis**
6. **Dermatofibroma**
7. **Vascular Lesion**
8. **Squamous Cell Carcinoma**

---

## Objectives
1. Develop a robust, scalable system for skin lesion classification.
2. Explore the use of deep learning models such as **EfficientNetB2**, **EfficientNetB3**, and **ResNet-50**.
3. Optimize computational efficiency using parallelism and GPU acceleration.
4. Achieve high accuracy and reliability while addressing challenges like class imbalance and computational complexity.

---

## Methodology

### **1. Data Preprocessing**
1. **Image Resizing and Normalization**:
   - Resize images to 224x224 pixels to match model input requirements.
   - Normalize pixel intensities to the range [0, 1].
   - Use ImageNet mean and standard deviation for consistent scaling.

2. **Data Augmentation**:
   - Techniques such as random rotations, horizontal/vertical flipping, and cropping were applied to enhance model generalization and address class imbalance.

3. **Parallelization**:
   - GPU-accelerated preprocessing pipeline for faster data loading and augmentation.

### **2. Model Selection**
The project employs four deep learning architectures:
- **Basic CNN**: A baseline model to establish performance benchmarks.
- **EfficientNetB2**: Optimized for computational efficiency with minimal trade-offs in accuracy.
- **EfficientNetB3**: A more advanced version, providing higher accuracy.
- **ResNet-50**: Known for its residual connections, enabling deeper network training.

### **3. Training Optimization**
1. **Parallel Processing**:
   - GPU acceleration for parallelized data loading and batch processing.

2. **Hyperparameter Tuning**:
   - Optimized learning rate, batch size, and dropout rates using grid search and cross-validation.

3. **Loss Function and Optimizer**:
   - Binary cross-entropy as the loss function.
   - Adam optimizer for efficient gradient descent and convergence.

4. **Validation**:
   - An 80:20 split between training and validation datasets.
   - Metrics: Accuracy, precision, recall, and confusion matrix.

### **4. Parallelism Impact**
| Model            | Without Parallelism | With Parallelism |
|-------------------|---------------------|------------------|
| Basic CNN        | 314s                | 630s             |
| EfficientNetB2   | 192s                | 108s             |
| EfficientNetB3   | 383s                | 183s             |
| ResNet-50        | 162s                | 162s             |

---

## Results
### **Performance Metrics**
| Model            | Training Accuracy | Validation Accuracy |
|-------------------|-------------------|---------------------|
| Basic CNN        | 0.93              | 0.53                |
| EfficientNetB2   | 0.82              | 0.81                |
| EfficientNetB3   | 0.98              | 0.82                |
| ResNet-50        | 0.96              | 0.82                |

### Key Observations:
1. EfficientNet models achieved the best balance between accuracy and computational efficiency.
2. ResNet-50 exhibited robust generalization but required more epochs for convergence.
3. Basic CNN faced significant overfitting, making it unsuitable for production-grade applications.

---

## Conclusion
This project successfully demonstrates the use of deep learning models for automated skin lesion classification. By leveraging parallelism and GPU acceleration, the system achieves high computational efficiency without compromising accuracy. The integration of advanced architectures like EfficientNet and ResNet makes this system a valuable tool for dermatological decision-making.

---

## How to Run
1. Clone the repository.
2. Set up the environment using the provided `requirements.txt`.
3. Train the models by running the corresponding scripts for each architecture.
4. View the results and visualizations in the `outputs` folder.

---

For further details, refer to the full project report included in this repository.
