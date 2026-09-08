# Pediatric Pneumonia Detection from Chest X-Ray Images

**Problem Set 01 --- Convolutional Neural Network (CNN) for Pediatric
Chest X-Ray Classification**

## Overview

This project develops a **Convolutional Neural Network (CNN)** to
classify pediatric chest X-ray images into two categories:

-   `NORMAL`
-   `PNEUMONIA`

The task is based on anterior-posterior (AP) chest X-ray images from
pediatric patients aged approximately 1--5 years. The dataset is
organized into `train`, `val`, and `test` directories, with separate
class folders for `NORMAL` and `PNEUMONIA`.

The complete implementation is provided in the accompanying
Jupyter/Google Colab notebook:

> `Problem_1.ipynb`

The notebook uses **TensorFlow/Keras** and is configured to take
advantage of a GPU when available.

------------------------------------------------------------------------

## Project Objectives

The main objectives of this assignment are to:

1.  Load and inspect the chest X-ray dataset.
2.  Examine the distribution of the two classes.
3.  Visualize representative X-ray images.
4.  Prepare image datasets for CNN training.
5.  Address class imbalance using class weights.
6.  Apply medically sensible image augmentation.
7.  Build and train a custom CNN.
8.  Evaluate the model using clinically relevant classification metrics.
9.  Analyze the confusion matrix and ROC curve.
10. Discuss the importance of sensitivity/recall for pneumonia
    detection.

------------------------------------------------------------------------

## Dataset

The dataset contains **5,863 JPEG chest X-ray images** according to the
assignment description.

The expected directory structure is:

``` text
dataset/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### Dataset distribution observed in the notebook

  Split          Normal   Pneumonia   Total
  ------------ -------- ----------- -------
  Train           1,362       3,932   5,294
  Validation          8           8      16
  Test              234         393     627

The supplied validation directory contains only 16 images. Therefore,
the notebook automatically creates an **80/20 training-validation split
from the original training directory**.

This produces:

-   **4,236 images** for training
-   **1,058 images** for validation
-   **627 images** retained as the independent test set

This handling is implemented automatically in the notebook.

------------------------------------------------------------------------

## Technologies and Libraries

The project was implemented using:

-   Python
-   TensorFlow / Keras
-   NumPy
-   Pandas
-   Matplotlib
-   Seaborn
-   Pillow (PIL)
-   Scikit-learn
-   Google Colab
-   GPU acceleration

The notebook was executed with:

``` text
TensorFlow: 2.20.0
Hardware: NVIDIA T4 GPU
```

------------------------------------------------------------------------

## Methodology

The overall workflow is:

``` text
Chest X-Ray Dataset
        │
        ▼
Dataset Discovery & Loading
        │
        ▼
Class Distribution Analysis
        │
        ▼
Image Visualization
        │
        ▼
Image Resizing → 224 × 224
        │
        ▼
Training / Validation Preparation
        │
        ▼
Class Weighting
        │
        ▼
Data Augmentation
        │
        ▼
CNN Model
        │
        ▼
Model Training
        │
        ▼
Best Model Checkpoint
        │
        ▼
Test Set Evaluation
        │
        ├── Classification Report
        ├── Confusion Matrix
        └── ROC Curve
```

------------------------------------------------------------------------

## Data Preprocessing

All images are resized to:

``` text
224 × 224 pixels
```

The images are loaded using TensorFlow's
`image_dataset_from_directory()` with binary labels.

Pixel values are rescaled from the original range to approximately:

``` text
0–1
```

using:

``` python
layers.Rescaling(1./255)
```

The notebook also uses caching and prefetching with TensorFlow's
`tf.data` pipeline to improve input performance.

------------------------------------------------------------------------

## Class Imbalance

The training data is substantially imbalanced:

-   Normal: **1,362**
-   Pneumonia: **3,932**

To reduce the effect of this imbalance, the model uses class weights.

The calculated weights are approximately:

``` text
NORMAL     = 1.9435
PNEUMONIA  = 0.6732
```

This gives the minority `NORMAL` class greater weight during
optimization rather than allowing the majority class to dominate the
learning process.

------------------------------------------------------------------------

## Data Augmentation

The CNN applies lightweight augmentation designed for chest X-ray
images:

``` python
RandomFlip("horizontal")
RandomRotation(0.05)
RandomZoom(0.05)
```

A **vertical flip is intentionally not used**, because changing the
vertical orientation of a chest X-ray would produce an anatomically
unrealistic image.

The augmentation strategy therefore attempts to introduce useful
variation while preserving the overall anatomical structure of the
chest.

------------------------------------------------------------------------

## CNN Architecture

A custom CNN named:

``` text
Pediatric_Pneumonia_CNN
```

was implemented.

### Architecture

``` text
Input: 224 × 224 × 3
        │
        ▼
Data Augmentation
        │
        ▼
Rescaling
        │
        ▼
Conv2D: 32 filters
Batch Normalization
ReLU
Max Pooling
        │
        ▼
Conv2D: 64 filters
Batch Normalization
ReLU
Max Pooling
        │
        ▼
Conv2D: 128 filters
Batch Normalization
ReLU
Max Pooling
        │
        ▼
Conv2D: 256 filters
Batch Normalization
ReLU
Max Pooling
        │
        ▼
Global Average Pooling
        │
        ▼
Dense: 128 neurons
Batch Normalization
Dropout: 0.4
        │
        ▼
Dense: 1 neuron
Sigmoid Activation
        │
        ▼
NORMAL / PNEUMONIA
```

### Model size

The network contains:

-   **423,873 total parameters**
-   **422,657 trainable parameters**
-   **1,216 non-trainable parameters**

------------------------------------------------------------------------

## Model Compilation

The model is compiled using:

``` python
Adam(learning_rate=1e-3)
```

with:

``` text
Loss       : Binary Cross-Entropy
Accuracy   : Classification accuracy
Precision  : Precision
Recall     : Recall / Sensitivity
AUC        : ROC-AUC
```

------------------------------------------------------------------------

## Training Strategy

The model is configured for a maximum of:

``` text
25 epochs
```

with a batch size of:

``` text
64
```

Three callbacks are used:

### 1. Early Stopping

Training stops when validation loss fails to improve for six consecutive
epochs, and the best weights are restored.

### 2. ReduceLROnPlateau

The learning rate is reduced when validation loss stops improving.

### 3. Model Checkpoint

The best model based on validation loss is saved as:

``` text
best_pneumonia_cnn.keras
```

The recorded training run stopped early at **epoch 17**, restoring the
best weights from **epoch 11**.

------------------------------------------------------------------------

## Training Results

The best recorded validation performance occurred at epoch 11:

  Metric        Validation Result
  ----------- -------------------
  Accuracy                 0.9783
  Precision                0.9863
  Recall                   0.9851
  AUC                      0.9970
  Loss                     0.0642

The model's training metrics continued to improve during training, while
validation performance fluctuated considerably across some epochs. Early
stopping and checkpointing were therefore important for retaining the
best-performing weights.

------------------------------------------------------------------------

## Test Set Results

The final restored model was evaluated on the test set containing **627
images**.

### Classification Report

  Class                    Precision   Recall     F1-score   Support
  ---------------------- ----------- -------- ------------ ---------
  NORMAL                      0.9545   0.3590       0.5217       234
  PNEUMONIA                   0.7217   0.9898       0.8348       393
  **Overall Accuracy**                          **0.7544**   **627**

Additional averages:

  Metric        Macro Avg   Weighted Avg
  ----------- ----------- --------------
  Precision        0.8381         0.8086
  Recall           0.6744         0.7544
  F1-score         0.6783         0.7179

The notebook also generates:

-   A test-set confusion matrix
-   A ROC curve
-   ROC-AUC calculation

The numerical ROC-AUC value from the final test evaluation is calculated
in the notebook but is not printed in the saved notebook output, so it
is intentionally not reported here.

------------------------------------------------------------------------

## Clinical Interpretation

For this healthcare classification problem, **recall for the `PNEUMONIA`
class is particularly important**.

A false negative means that an X-ray belonging to a pneumonia case is
incorrectly classified as normal. In a clinical setting, this could
potentially delay further assessment or treatment.

The test results show:

``` text
PNEUMONIA Recall = 98.98%
```

This indicates that the model detected most of the pneumonia cases in
the test set.

However, this comes with a trade-off. The recall for the `NORMAL` class
is only:

``` text
NORMAL Recall = 35.90%
```

Therefore, many normal cases are classified as pneumonia.

This is an important limitation and demonstrates why **accuracy alone is
not sufficient for evaluating medical image classification models**.

------------------------------------------------------------------------

## Key Findings

### Strengths

-   Custom CNN successfully learns meaningful patterns from chest X-ray
    images.
-   High pneumonia recall on the held-out test set.
-   Class imbalance is explicitly addressed through class weighting.
-   Medical-domain-aware augmentation is used.
-   Batch normalization and dropout are included to improve training
    stability and generalization.
-   Early stopping prevents unnecessary training.
-   Multiple evaluation metrics are reported rather than relying only on
    accuracy.
-   Confusion matrix and ROC analysis provide additional insight into
    model behavior.

### Limitations

-   The validation directory supplied with the dataset contains only 16
    images, so the notebook creates a new 80/20 validation split from
    the training directory.
-   The test results show a substantial difference between pneumonia
    recall and normal recall.
-   The overall test accuracy of **75.44%** indicates that the model is
    not sufficiently reliable for direct clinical deployment.
-   The model is a custom CNN rather than a clinically validated
    diagnostic system.
-   The dataset is limited to the population and imaging conditions
    represented in the source data.
-   Performance on external hospitals, scanners, age groups,
    populations, or acquisition protocols may differ.

------------------------------------------------------------------------

## Important Healthcare Disclaimer

This project is intended for **educational and research purposes only**.

The trained model should **not be used as a standalone diagnostic tool**
and should not replace assessment by qualified healthcare professionals.

A model intended for real clinical use would require substantially more
validation, including external validation, careful assessment of false
negatives and false positives, calibration, subgroup analysis, clinical
review, data governance, and appropriate regulatory approval.

------------------------------------------------------------------------

## How to Run the Project

### Option 1 --- Google Colab

1.  Open `Problem_1.ipynb` in Google Colab.
2.  Upload or place the dataset in Google Drive.
3.  Make sure the dataset contains:

``` text
train/NORMAL
train/PNEUMONIA
val/NORMAL
val/PNEUMONIA
test/NORMAL
test/PNEUMONIA
```

4.  Run the notebook cells sequentially.
5.  When prompted, allow Google Colab to access Google Drive.
6.  For faster training, select:

``` text
Runtime → Change runtime type → T4 GPU
```

The notebook automatically searches Google Drive for an appropriate
dataset directory or ZIP archive.

### Dataset path

The recorded run located the dataset at:

``` text
/content/drive/MyDrive/problem 1/New folder (2)
```

For another environment, the `BASE_DIR` variable can be manually changed
to the location of the dataset.

------------------------------------------------------------------------

## Expected Outputs

Running the notebook produces:

1.  TensorFlow and hardware information
2.  Dataset distribution table
3.  Class distribution visualization
4.  Sample chest X-ray visualizations
5.  CNN architecture summary
6.  Training and validation curves
7.  Classification report
8.  Confusion matrix
9.  ROC curve
10. Saved best model:

``` text
best_pneumonia_cnn.keras
```

------------------------------------------------------------------------

## Project Structure

A recommended project structure is:

``` text
Pediatric-Pneumonia-CNN/
│
├── Problem_1.ipynb
├── README.md
│
├── dataset/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
│
└── best_pneumonia_cnn.keras
```

The dataset itself is not included in this project repository because
medical imaging datasets may have separate distribution and usage
requirements.

------------------------------------------------------------------------

## Reproducibility

A random seed of:

``` python
SEED = 42
```

is used for TensorFlow and NumPy.

This helps make the data splitting and training process more
reproducible, although complete reproducibility can still vary depending
on hardware, TensorFlow version, GPU operations, and execution
environment.

------------------------------------------------------------------------

## Conclusion

This project demonstrates an end-to-end CNN workflow for pediatric
pneumonia classification from chest X-ray images.

The model achieved a **75.44% test accuracy** and a **98.98% recall for
pneumonia** on the evaluated test set. The results highlight the
usefulness of CNNs for medical image classification while also
demonstrating an important healthcare trade-off: maximizing pneumonia
sensitivity can increase false-positive classifications of normal
images.

The findings reinforce the importance of evaluating medical AI systems
using **precision, recall, F1-score, confusion matrices, and ROC-AUC**,
rather than relying solely on overall accuracy.

For future improvement, the project could be extended with transfer
learning, stronger regularization, improved validation design, threshold
optimization, external validation, explainability techniques such as
Grad-CAM, and systematic clinical error analysis.
