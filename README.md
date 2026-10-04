# PCA + Logistic Regression — Breast Cancer Classification

## Project Overview

This project demonstrates **Principal Component Analysis (PCA)** for dimensionality reduction and then uses **Logistic Regression** for binary classification on the Breast Cancer Wisconsin dataset from scikit-learn.

The notebook follows a learning-focused ML workflow: EDA → train/test split → scaling → PCA → Logistic Regression → classification evaluation → model comparison.

### Main workflow

```text
Breast Cancer Dataset
        ↓
Exploratory Data Analysis
        ↓
Train / Test Split
        ↓
StandardScaler
        ↓
PCA
        ↓
Select 10 Principal Components
        ↓
Logistic Regression
        ↓
Predictions + Probabilities
        ↓
Confusion Matrix
        ↓
Classification Metrics
        ↓
ROC-AUC + ROC Curve
        ↓
Compare with Logistic Regression using all 30 features
```

## Dataset

The project uses `load_breast_cancer()` from `sklearn.datasets`.

- **569 observations**
- **30 numerical input features**
- **2 target classes**
- No missing values in the input features

Target encoding:

- `0` → Malignant
- `1` → Benign

> In the notebook's default classification metrics, class `1` (benign) is treated as the positive class.

## Why PCA?

The dataset contains **30 features**. PCA reduces dimensionality while retaining as much variance as possible.

PCA does not simply select 10 original columns. It creates **new features called principal components**.

The project studies principal components, explained variance, cumulative explained variance, dimensionality reduction, and PCA visualization.

## Exploratory Data Analysis

The notebook checks dataset shape, data types, feature statistics, missing values, target class distribution, and feature scales.

The features have very different numerical scales. Since PCA is variance-based, feature scaling is performed before PCA.

## Train/Test Split

The data is split into:

- **80% training data**
- **20% testing data**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

`stratify=y` helps preserve approximately the same class proportions in the training and test sets.

The split is performed before fitting the scaler so that information from the test set does not influence preprocessing.

## Feature Scaling

`StandardScaler` is used before PCA:

```python
scaler = StandardScaler()
X_train_trans = scaler.fit_transform(X_train)
X_test_trans = scaler.transform(X_test)
```

Scaling matters because PCA searches for directions of maximum variance. A feature with a much larger numerical scale can otherwise dominate the variance calculation.

**Important:** Scaling does not remove outliers; it changes the numerical scale of the features.

## PCA Analysis

### Two-component PCA

The first two components capture approximately:

- **PC1:** 44.41%
- **PC2:** 18.94%
- **PC1 + PC2:** 63.36%

This gives a convenient 2D visualization, but it does not retain all original variance.

### Component Selection

Cumulative explained variance is used to decide how many components to retain.

Using a 95% variance-retention target:

- **10 principal components** retain approximately **95.27%** of the training variance.

The classification stage therefore uses 10 PCA components.

## PCA Visualization

The first two principal components are plotted:

- X-axis → PC1
- Y-axis → PC2
- Point colors → target class

This helps visually inspect whether the classes appear separated in the reduced feature space.

# Logistic Regression after PCA

After the PCA stage, Logistic Regression is used as the supervised classification model.

```text
PCA → dimensionality reduction
Logistic Regression → classification
```

PCA learns from `X` only, while Logistic Regression learns the relationship between features and target `y`.

```python
from sklearn.linear_model import LogisticRegression

lor_pca = LogisticRegression(max_iter=1000)
lor_pca.fit(X_train_pca, y_train)
```

## Predictions and Probabilities

Class predictions:

```python
y_pred_pca = lor_pca.predict(X_test_pca)
```

Class-1 probabilities:

```python
y_pred_proba_pca = lor_pca.predict_proba(X_test_pca)[:, 1]
```

The default classification threshold is 0.5.

## Confusion Matrix

```text
                 Predicted
                0        1

Actual  0      TN       FP
        1      FN       TP
```

- **TN** → True Negative
- **FP** → False Positive
- **FN** → False Negative
- **TP** → True Positive

For this dataset, `0 = malignant` and `1 = benign`, so the positive-class metrics should be interpreted accordingly.

## Classification Metrics

The notebook calculates:

- **Accuracy** — overall fraction of correct predictions
- **Precision** — among samples predicted as the positive class, how many were actually positive
- **Recall** — among actual positive samples, how many were correctly identified
- **F1-score** — harmonic mean of precision and recall
- **ROC-AUC** — how well the model ranks the two classes across probability thresholds

A classification report is also generated with precision, recall, F1-score, and support.

## ROC Curve

The ROC curve plots:

```text
True Positive Rate / Recall
          vs
False Positive Rate
```

ROC-AUC close to 1 indicates strong class separation, while an AUC around 0.5 indicates roughly random ranking.

ROC-AUC is calculated from predicted probabilities rather than only the final 0/1 predictions.

## Comparing PCA vs Original Features

The notebook compares:

1. Logistic Regression using all **30 standardized original features**.
2. Logistic Regression using **10 PCA components**.

The comparison includes:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The model using all 30 standardized features performs slightly better than the 10-component PCA model on the selected train/test split.

This does **not** mean PCA failed. PCA was used for dimensionality reduction while retaining approximately 95% of the training variance. The remaining variance can still contain information useful for classification.

The key lesson is:

- For dimensionality reduction → explained variance is important.
- For supervised prediction → classification performance should also be evaluated when choosing the number of components.

## Key Concepts Learned

### PCA

- Dimensionality reduction
- Variance
- Principal components
- Explained variance
- Cumulative explained variance
- Feature scaling before PCA
- PCA visualization

### Logistic Regression

- Binary classification
- Predicted probabilities
- Classification threshold
- Logistic Regression after dimensionality reduction

### Classification Evaluation

- Confusion matrix
- True Positive / True Negative
- False Positive / False Negative
- Accuracy
- Precision
- Recall
- F1-score
- ROC curve
- ROC-AUC

### ML Workflow

- Exploratory Data Analysis
- Train/test split
- Preventing preprocessing leakage
- Standardization
- Dimensionality reduction
- Model training
- Prediction
- Evaluation
- Model comparison

## Project Structure

```text
PCA-Logistic-Regression/
│
├── PCA_Project_LogisticRegression_Completed.ipynb
└── README.md
```

The dataset does not need to be downloaded separately because it is loaded directly from scikit-learn.

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn
```

The notebook was developed using Python 3.12.

## How to Run

1. Clone or download the repository.
2. Open the notebook in Jupyter Notebook, JupyterLab, VS Code, or another compatible environment.
3. Run the cells from top to bottom.

```bash
jupyter notebook
```

Then open `PCA_Project_LogisticRegression_Completed.ipynb`.

## Learning Goal

The main purpose of this project is to understand how PCA and Logistic Regression work together rather than treating them as black-box algorithms.

```text
PCA
Unsupervised → learns from X only
        ↓
Reduced features
        ↓
Logistic Regression
Supervised → learns from X and y
        ↓
Predicted classes
        ↓
Classification metrics
```

## Author

**Rahul**

This project was created as part of a hands-on Machine Learning learning journey.
