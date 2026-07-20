# Loan Approval Prediction Model

A machine learning project developed in Python to predict whether a loan application should be approved based on previous application outcomes.

## Project Overview

The dataset contains applicant information such as age, income and outgoings. The project cleans the supplied data, creates affordability-related features and compares Logistic Regression with Random Forest.

Logistic Regression was selected as the final model because it provided slightly better overall performance while remaining easier to interpret for financial decision-making.

## Features Used

- Age
- Income
- Outgoings
- Disposable income
- Outgoings-to-income ratio

Applicant ID, gender and postcode were excluded from prediction because they were either irrelevant identifiers or presented privacy, fairness and discrimination concerns.

## Data Preparation

The preprocessing process includes:

- Removing duplicate records
- Converting invalid income values such as `nil` into missing values
- Identifying unrealistic outgoings
- Median imputation for missing numerical data
- Creating affordability-based features
- Standardising features for Logistic Regression
- Using a stratified 80:20 train-test split
- Preventing data leakage with a scikit-learn pipeline

## Models Compared

1. Logistic Regression
2. Random Forest

## Model Performance

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.654 | 0.661 | 0.837 | 0.739 | 0.701 |
| Random Forest | 0.649 | 0.671 | 0.782 | 0.723 | 0.690 |

Logistic Regression was selected as the final model due to its higher F1-score, recall and ROC-AUC, alongside its greater interpretability.

## Visualisations

The program generates:

- Confusion matrix
- ROC curve comparison
- Permutation feature importance
- Model performance table

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Installation

Clone the repository:

```bash
git clone https://github.com/Bikin-mainali/Loan_Approval_Model.git
cd Loan_Approval_Model


## Install the required package
pip install pandas numpy scikit-learn matplotlib seaborn

## Running the model
python loan_approval_model.py

## Project Structure
loan-approval-model/
├── loan_approval_model.py
├── upload/
│   └── previousApplicants.csv
├── model_outputs/
│   ├── model_metrics.csv
│   ├── confusion_matrix.png
│   ├── roc_comparison.png
│   └── permutation_importance.png
└── README.md

Author

Bikin Mainali
MSc Applied Artificial Intelligence student
University of Chester
