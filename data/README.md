# HAM10000 Dataset Setup

This project uses the **HAM10000 (Human Against Machine with 10,000 training images)** dataset for multi-class skin lesion classification.

The dataset is **not stored directly in this GitHub repository** because of its large size.

## Dataset Overview

HAM10000 contains:

- 10,015 dermoscopic images
- Metadata for each image
- 7 diagnostic skin lesion classes
- Multiple images that may belong to the same physical lesion

The target label used in this project is:

```text
dx
```

The seven classes are:

| Code | Diagnosis |
|---|---|
| `akiec` | Actinic keratoses / Intraepithelial carcinoma |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic nevi |
| `vasc` | Vascular lesions |

## Expected Directory Structure

After downloading HAM10000, place the dataset inside:

```text
data/raw/
```

The expected structure is:

```text
data/
├── README.md
├── raw/
│   ├── HAM10000_metadata.csv
│   ├── HAM10000_images_part_1/
│   │   ├── ISIC_0024306.jpg
│   │   ├── ISIC_0024307.jpg
│   │   └── ...
│   └── HAM10000_images_part_2/
│       ├── ISIC_0032258.jpg
│       ├── ISIC_0032259.jpg
│       └── ...
│
└── processed/
    ├── cleaned_metadata.csv
    ├── train.csv
    ├── val.csv
    └── test.csv
```

The exact generated filenames inside `data/processed/` may depend on the preprocessing pipeline, but the main train, validation, and test split files should remain consistent across all experiments.

## Raw Data

The `data/raw/` directory contains the original HAM10000 files.

Typical files include:

```text
HAM10000_metadata.csv
HAM10000_images_part_1/
HAM10000_images_part_2/
```

Raw data should not be manually modified.

## Processed Data

The `data/processed/` directory contains files generated during data cleaning and splitting.

Typical outputs include:

```text
cleaned_metadata.csv
train.csv
val.csv
test.csv
```

These files are generated from the raw HAM10000 dataset.

The project uses a fixed train/validation/test split so that every model is evaluated on exactly the same data.

## Important: Lesion-Aware Splitting

HAM10000 may contain multiple images belonging to the same physical lesion.

For this reason, the project does not rely on a simple random image-level split.

Instead, splitting is performed using `lesion_id` so that images belonging to the same lesion remain in the same partition.

This helps prevent data leakage between:

```text
Training
Validation
Testing
```

Once the split has been generated, it should be reused for all models.

Do not independently regenerate a different split for each model.

## Recommended Execution Order

The dataset pipeline is organized through the notebooks:

```text
01_data_understanding.ipynb
02_data_cleaning.ipynb
03_data_split.ipynb
04_preprocessing_augmentation.ipynb
```

Recommended order:

```text
Raw HAM10000 Dataset
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
Lesion-Aware Data Split
        ↓
Image Preprocessing
        ↓
Data Augmentation
        ↓
Model Training
```

## Image Preprocessing

Images are resized to:

```text
224 × 224
```

and converted to RGB tensors with shape:

```text
3 × 224 × 224
```

The project uses ImageNet normalization:

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

Training images may additionally use random augmentation.

Validation and test images use deterministic preprocessing only.

## Data Augmentation

Data augmentation is applied only to the training set.

Typical transformations include:

- Random horizontal flipping
- Random vertical flipping
- Random rotation
- Random affine transformations
- Color or intensity transformations

Validation and test data must not use random augmentation.

## GitHub Data Policy

Dataset files are excluded from GitHub through `.gitignore`.

The repository keeps only the folder structure:

```text
data/raw/.gitkeep
data/processed/.gitkeep
```

Large image files, CSV files, and generated processed data should remain local.

This prevents the HAM10000 dataset from being accidentally committed to the repository.

## Reproducibility

For consistent model comparison:

- Use the same fixed random seed.
- Use the same lesion-aware split.
- Reuse the same `train.csv`, `val.csv`, and `test.csv`.
- Use the same preprocessing configuration.
- Use the same class-to-index mapping.

The class order used in this project is:

```text
akiec: 0
bcc:   1
bkl:   2
df:    3
mel:   4
nv:    5
vasc:  6
```

All models should follow this same mapping.