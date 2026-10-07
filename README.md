# Skin Lesion Classification using Deep Learning

## 1. Project Overview

Skin lesion classification is an important application of deep learning in medical image analysis. Different types of skin lesions can exhibit very similar visual characteristics, making manual diagnosis challenging and highly dependent on clinical expertise.

This project develops a **deep learning-based image classification pipeline** for automatically classifying dermoscopic skin lesion images into multiple diagnostic categories.

The project uses the **HAM10000 (Human Against Machine with 10,000 training images)** dataset and compares different Convolutional Neural Network (CNN) approaches, ranging from simple CNN architectures built from scratch to more advanced CNN models and transfer learning.

The main objective is not only to achieve good classification performance, but also to systematically study how different CNN architectures affect the performance of a multi-class medical image classification task.

---

## 2. Project Objectives

The main objectives of this project are:

1. Explore and understand the HAM10000 skin lesion dataset.
2. Clean and validate the metadata and image files.
3. Split the dataset into training, validation, and testing sets while preventing data leakage.
4. Preprocess dermoscopic images and apply data augmentation.
5. Design and train a **Simple CNN** model from scratch.
6. Design and train a **Complex CNN** model using deeper CNN blocks.
7. Apply **Transfer Learning** using a pretrained CNN architecture.
8. Perform **Fine-Tuning** on the pretrained model.
9. Evaluate and compare all models using appropriate classification metrics.
10. Analyze model errors and limitations.

---

## 3. Dataset

This project uses the **HAM10000 dataset (Human Against Machine with 10,000 training images)**.

The dataset contains:

- **10,015 dermoscopic images**
- Metadata associated with each image
- Seven diagnostic skin lesion categories
- Multiple images may belong to the same physical lesion

The metadata file contains information such as:

- `lesion_id`
- `image_id`
- `dx`
- `dx_type`
- `age`
- `sex`
- `localization`

The target label used for classification is:

```text
dx
```

### Skin Lesion Classes

The classification task contains seven classes:

| Code | Diagnosis |
|------|-----------|
| `akiec` | Actinic keratoses / Intraepithelial carcinoma |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic nevi |
| `vasc` | Vascular lesions |

One important characteristic of HAM10000 is its **significant class imbalance**, with melanocytic nevi (`nv`) representing the majority of images while several other lesion types contain substantially fewer samples.

This imbalance is considered during model training and evaluation.

---

## 4. Project Pipeline

The project follows the following end-to-end pipeline:

```text
HAM10000 Dataset
        │
        ▼
Data Understanding
        │
        ▼
Data Cleaning & Validation
        │
        ▼
Lesion-aware Data Splitting
        │
        ▼
Image Preprocessing
        │
        ▼
Data Augmentation
        │
        ├─────────────────────────────┐
        │                             │
        ▼                             ▼
  Simple CNN                    Complex CNN
        │                             │
        └──────────────┬──────────────┘
                       │
                       ▼
              Transfer Learning
                       │
                       ▼
                  Fine-Tuning
                       │
                       ▼
               Model Evaluation
                       │
                       ▼
                 Error Analysis
```

---

# 5. Data Processing

## 5.1 Data Understanding

The first stage explores the HAM10000 metadata and image dataset.

The analysis includes:

- Dataset dimensions
- Metadata structure
- Skin lesion class distribution
- Missing values
- Duplicate images
- Number of unique lesions
- Relationship between `lesion_id` and `image_id`
- Distribution of demographic and lesion-related attributes

A particularly important observation is that multiple images can correspond to the **same lesion**.

Therefore, randomly splitting individual images could introduce **data leakage**, because images of the same lesion might appear in both training and testing sets.

---

## 5.2 Data Cleaning

The cleaning pipeline validates both metadata and image files.

Main cleaning steps include:

- Checking required metadata columns
- Validating image IDs
- Matching metadata records with existing image files
- Removing invalid or missing image references
- Handling missing metadata where necessary
- Creating a consistent dataset for later preprocessing

The cleaned metadata serves as the main input for dataset splitting and model training.

---

## 5.3 Data Splitting

The dataset is divided into:

```text
Training Set
Validation Set
Testing Set
```

Instead of performing a completely random image-level split, the splitting process considers `lesion_id`.

Images belonging to the same lesion are kept within the same data partition.

This prevents the model from seeing images of the same physical lesion during both training and evaluation.

A fixed random seed is used to make the split **reproducible** across experiments.

The generated split is reused by all models so that the comparison between different architectures remains fair.

---

# 6. Image Preprocessing

Before images are passed into the CNN models, they are transformed into a consistent format.

The preprocessing pipeline includes:

### Image Loading

Images are loaded from the two original HAM10000 image directories:

```text
HAM10000_images_part_1
HAM10000_images_part_2
```

### Image Resizing

All images are resized to:

```text
224 × 224
```

This provides a consistent input size for both custom CNN architectures and pretrained models.

### Tensor Conversion

Images are converted into PyTorch tensors.

The final input shape is:

```text
3 × 224 × 224
```

where:

- `3` = RGB channels
- `224` = image height
- `224` = image width

### Normalization

Pixel values are normalized before being passed into the neural network.

Normalization helps stabilize the optimization process and improves training convergence.

---

# 7. Data Augmentation

Data augmentation is applied only to the **training dataset**.

The purpose of augmentation is to artificially increase image diversity and reduce overfitting.

Typical augmentation operations include transformations such as:

- Random horizontal flipping
- Random vertical flipping
- Random rotations
- Random affine transformations
- Color/intensity transformations
- Resizing and normalization

Validation and testing images do **not** use random augmentation.

They only use deterministic preprocessing to ensure consistent evaluation.

The general transformation strategy is therefore:

```text
Training:
Image
 → Resize
 → Random Augmentation
 → Tensor
 → Normalize

Validation / Testing:
Image
 → Resize
 → Tensor
 → Normalize
```

---

# 8. Model 1 — Simple CNN

The first model is a custom Convolutional Neural Network designed and trained **from scratch**.

It provides a baseline for evaluating more advanced architectures.

The model produces seven output logits corresponding to the seven skin lesion classes.

No Softmax layer is included directly in the architecture because PyTorch's `CrossEntropyLoss` internally handles the required log-softmax operation.

The Simple CNN contains approximately **620K trainable parameters**.

### Purpose

The Simple CNN acts as the baseline model.

Its results will later be compared against:

```text
Simple CNN
    vs
Complex CNN
    vs
Transfer Learning
    vs
Fine-Tuned Model
```

---

# 9. Model 2 — Complex CNN

The second model extends the Simple CNN by using a deeper architecture composed of multiple CNN blocks.

The goal is to increase the model's ability to learn hierarchical image features.

Early convolutional layers primarily capture simple visual patterns such as:

- Edges
- Colors
- Local textures

Deeper layers can learn more complex lesion characteristics such as:

- Irregular lesion boundaries
- Pigmentation patterns
- Texture structures
- Shape characteristics
- Higher-level visual representations

Multiple CNN blocks are stacked to progressively increase feature depth while reducing spatial dimensions.

Regularization techniques such as **Dropout** and **Batch Normalization** are used to improve training stability and reduce overfitting.

The Complex CNN is also trained from scratch, allowing direct comparison with the simpler baseline architecture.

---

# 10. Transfer Learning

Training a deep CNN from scratch can be challenging when working with a relatively small medical image dataset.

To address this limitation, the project also investigates **Transfer Learning**.

A CNN pretrained on a large image dataset is used as a feature extractor.

The original classification layer is replaced with a new classifier for the seven HAM10000 lesion classes.

During the initial transfer-learning stage, most or all pretrained backbone layers are frozen.

Only the newly added classification layers are trained.

This allows the project to reuse visual features learned from a much larger dataset.

---

# 11. Fine-Tuning

After transfer learning, the pretrained model is further optimized through **fine-tuning**.

Selected layers of the pretrained backbone are unfrozen and trained using a smaller learning rate.

Fine-tuning allows the pretrained features to adapt more specifically to dermoscopic skin lesion images.

---

# 12. Training Strategy

The models are implemented using **PyTorch**.

### Loss Function

The primary classification loss is:

```python
CrossEntropyLoss
```

This loss is suitable for the seven-class classification problem.

Class imbalance handling can also be incorporated into the training process using techniques such as class-weighted loss or sampling strategies.

### Optimization

Model parameters are updated using gradient-based optimization.

Training statistics are recorded for every epoch, including:

- Training loss
- Validation loss
- Training accuracy
- Validation accuracy

These statistics allow the learning behavior and possible overfitting of each architecture to be analyzed.

---

# 13. Model Evaluation

All trained models are evaluated using the same held-out test dataset.

Because HAM10000 is an imbalanced multi-class dataset, overall accuracy alone is not sufficient.

The project evaluates models using metrics including:

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1-score
- Weighted F1-score
- Confusion Matrix

Per-class metrics are especially important because a model can achieve relatively high overall accuracy while performing poorly on minority lesion classes.


# 14. Error Analysis

After model evaluation, incorrectly classified images are analyzed to understand common model failure cases.

The error analysis focuses on:

- Confusion between visually similar lesion categories
- Minority-class prediction errors
- False melanoma predictions
- Missed melanoma cases
- Difficult or ambiguous dermoscopic images
- Model confidence for incorrect predictions

This analysis provides additional insight beyond aggregate evaluation metrics.

---

# 15. Repository Structure


```text
skin-lesion-classification/
│
├── configs/
│   ├── simple_cnn.yaml
│   ├── complex_cnn.yaml
│   ├── transfer.yaml
│   ├── fine_tuning.yaml
│   ├── evaluation.yaml
│   └── error_analysis.yaml
│
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_data_split.ipynb
│   ├── 04_preprocessing_augmentation.ipynb
│   ├── 05_simple_cnn.ipynb
│   ├── 06_complex_cnn.ipynb
│   ├── 07_transfer_learning.ipynb
│   ├── 08_fine_tuning.ipynb
│   ├── 09_evaluation.ipynb
│   └── 10_error_analysis.ipynb
│
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── evaluation/
│
├── scripts/
├── tests/
├── outputs/
├── requirements.txt
└── README.md
```

The repository separates exploratory notebooks from reusable Python modules to make experiments easier to reproduce and maintain.

---

# 16. Technologies

The main technologies used in this project include:

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Scikit-learn
- Pillow
- OpenCV
- Albumentations
- Matplotlib
- Seaborn
- Jupyter Notebook
- YAML

---

# 17. Installation

Clone the repository:

```bash
git clone <repository-url>
cd skin-lesion-classification
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# 18. Dataset Setup

The HAM10000 dataset is not stored directly in this repository because of its size.

After downloading the dataset, place the original files inside:

```text
data/raw/
```

The expected structure should contain the metadata file and HAM10000 image folders.

Processed metadata, split information, and other generated files are stored under:

```text
data/processed/
```

---

# 19. Reproducibility

To make model comparisons reliable, the project uses a consistent experimental setup.

Important reproducibility practices include:

- Fixed random seeds
- Fixed train/validation/test split
- Lesion-aware dataset splitting
- Consistent image preprocessing
- Identical test data for all models
- Configuration files for individual experiments
- Separation between reusable source code and experiment notebooks

Once the dataset split has been generated, the same split should be reused rather than regenerated independently for each model.

This ensures that differences in model performance are caused primarily by model architecture and training strategy rather than differences in the evaluation data.

---


