# Problem Set 02 — Bank Marketing Prediction using Logistic Regression

## Overview

This project uses the **Bank Marketing Data Set** to build a binary classification model that predicts whether a bank customer will subscribe to a **term deposit**.

The classification target is:

- `yes` → customer subscribed to a term deposit
- `no` → customer did not subscribe

The model is implemented using **Logistic Regression** with preprocessing for numerical and categorical features.

## Dataset

The project uses the file:

```text
bank-full.csv
```

The dataset contains **45,211 records and 17 columns**. The CSV uses a semicolon (`;`) as its separator.

The target column is `y`.

## Project Objectives

The notebook performs the following steps:

1. Load and inspect the Bank Marketing dataset.
2. Explore the structure and distribution of the data.
3. Check for missing values and duplicate records.
4. Perform exploratory data analysis (EDA).
5. Separate input features and the target variable.
6. Convert the target from `yes/no` to `1/0`.
7. Split the data into training and testing sets using a stratified 80/20 split.
8. Standardize numerical features.
9. Apply One-Hot Encoding to categorical features.
10. Build a preprocessing and Logistic Regression pipeline.
11. Train the classification model.
12. Evaluate the model using multiple classification metrics.
13. Visualize the confusion matrix and ROC curve.
14. Interpret the learned Logistic Regression coefficients.
15. Demonstrate prediction for a new customer.

## Methodology

### 1. Data Preprocessing

The dataset contains both numerical and categorical variables. To prepare the data for Logistic Regression:

- Numerical features are **standardized** using `StandardScaler`.
- Categorical features are converted into numerical indicator variables using **One-Hot Encoding**.
- The target variable is encoded as:
  - `no = 0`
  - `yes = 1`

The preprocessing is included inside a single `scikit-learn` pipeline to avoid data leakage.

### 2. Train-Test Split

The dataset is divided into:

- **80% training data**
- **20% testing data**

A **stratified split** is used so the class distribution remains approximately consistent between the training and testing sets.

### 3. Logistic Regression

Logistic Regression is used because the target variable has two possible classes. The model produces a probability between 0 and 1 for the `yes` class.

Using the default threshold:

```text
Probability >= 0.50 → yes
Probability < 0.50  → no
```

## Evaluation Metrics

The model is evaluated using:

### Accuracy

Measures the overall percentage of correct predictions.

### Precision

Measures how many customers predicted as subscribers actually subscribed.

### Recall

Measures how many actual subscribers were correctly identified.

### F1-Score

The harmonic mean of precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between the two classes over different classification thresholds.

### Confusion Matrix

Shows the number of:

- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)

## Notebook Contents

The main notebook is:

```text
Problem_Set_02_Bank_Marketing_Logistic_Regression.ipynb
```

It contains:

- Dataset loading and inspection
- Descriptive statistics
- Missing-value and duplicate checks
- Exploratory visualizations
- Feature preparation
- Train-test splitting
- Data preprocessing
- Logistic Regression training
- Model predictions
- Model evaluation
- Confusion matrix
- ROC curve
- Coefficient interpretation
- Example customer prediction
- Final conclusion

## Requirements

The notebook requires Python 3 and the following libraries:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Install them with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## How to Run

1. Place the following files in the same directory:

```text
bank-full.csv
Problem_Set_02_Bank_Marketing_Logistic_Regression.ipynb
```

2. Open Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

3. Open:

```text
Problem_Set_02_Bank_Marketing_Logistic_Regression.ipynb
```

4. Run the notebook cells from top to bottom.

## Important Note About the Dataset

The target classes are imbalanced, with substantially more `no` responses than `yes` responses. Therefore, **accuracy alone is not sufficient** for judging model performance. Precision, recall, F1-score, ROC-AUC, and the confusion matrix should also be considered.

## Important Modeling Consideration

The `duration` variable represents the duration of the marketing contact. If the model is intended to make a prediction **before the marketing call takes place**, this variable may introduce deployment-time information leakage because its value is only known during or after the contact.

For a real-world deployment scenario, features that would not be available at prediction time should be removed or handled carefully.

## Expected Outcome

After running the notebook, the user will obtain:

- A trained Logistic Regression classifier
- Classification performance metrics
- A confusion matrix
- An ROC curve with ROC-AUC
- Important positive and negative model coefficients
- A probability-based prediction for a sample customer

## Author / Course

**Problem Set 02**  
**Topic:** Logistic Regression — Bank Marketing Prediction
