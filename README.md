# 💳 Credit Card Fraud Detection using Machine Learning
A complete end-to-end Machine Learning pipeline for detecting fraudulent credit card transactions using the ULB Credit Card Fraud Detection Dataset. This project handles severe class imbalance using SMOTE, trains multiple ML models, performs hyperparameter tuning, and generates detailed evaluation visualizations.

## Features

- Data loading and exploration
- Data preprocessing and feature scaling
- Class imbalance handling using SMOTE
- Multiple machine learning models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Linear SVM
- Hyperparameter tuning with GridSearchCV
- 5-Fold Cross Validation
- ROC Curve Analysis
- Precision-Recall Curve Analysis
- Confusion Matrix Visualization
- Feature Importance Analysis
- Threshold Optimization
- SMOTE Ablation Study

## Dataset
This project uses the Credit Card Fraud Detection Dataset provided by ULB.
Download the dataset from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Place the `creditcard.csv` file in the same directory as the Python script before running the project.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-Learn
- Imbalanced-Learn (SMOTE)
- Matplotlib
- Seaborn

## Installation
Clone the repository:
```bash
git clone https://github.com/kimayayaya/credit-card-fraud-detection.git
cd credit-card-fraud-detection
