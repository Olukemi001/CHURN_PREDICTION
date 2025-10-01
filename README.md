# CHURN_PREDICTION
# Project Overview
This project focuses on predicting customer churn for a telecommunications company using machine learning. Customer churn (when customers stop using a service) is a critical business metric, and accurately predicting it enables companies to take proactive retention measures.

The project implements and compares multiple machine learning models, with XGBoost selected as the production-ready model due to its superior performance in handling complex patterns in customer data.

# Business Objective
Primary Goal: Build a robust churn prediction model that can:

1. Identify customers at high risk of churning

2. Enable targeted retention campaigns

3. Reduce customer acquisition costs by improving retention

4. Provide insights into factors driving churn

# Dataset Description
The dataset contains 7,043 customer records with 56 features including:

## Target Variable
Churn: Binary indicator (1 = Churned, 0 = Not Churned)

Churn Rate: 26.54% (1,869 churned customers)

## Key Features
### Demographic Information
Age, Gender, Senior Citizen, Married, Dependents

Geographic data: Country, State, City, Zip Code, Latitude, Longitude

### Service & Usage Metrics
Monthly Charge, Total Charges, Total Revenue, Total Refunds

Total Extra Data Charges, Total Long Distance Charges

Avg Monthly GB Download, ARPU (Average Revenue Per User)

### Service Subscriptions
Phone Service, Multiple Lines, Internet Service, Internet Type

Online Security, Online Backup, Device Protection Plan

Premium Tech Support, Streaming Services (TV, Movies, Music)

Unlimited Data, Contract Type, Payment Method

### Customer Behavior & Engagement
Tenure in Months, Number of Referrals, Satisfaction Score

Payment Consistency, Service Bundling Score

Complaint Frequency, Engagement Index, CLTV (Customer Lifetime Value)

### Derived Features
Tenure Category, Age Group, Senior Flag

# Technical Implementation
## Data Preprocessing
### Feature Engineering
Leakage Prevention: Removed direct churn indicators:

Customer Status, Churn Label, Churn Reason, Churn Category, Churn Score

### Data Splitting
Train-Test Split: 80-20 split with stratification

Training set: 5,634 customers

Test set: 1,409 customers

Stratified sampling ensures consistent churn distribution (26.5% in both sets)

# Preprocessing Pipelines
Numerical Features (22 features)

Categorical Features (27 features)


# Model Architecture
1. Baseline Models
   
  a. Logistic Regression: Linear baseline for benchmarking

  b. Random Forest: Ensemble method for comparison

2. Production Model - XGBoost
   
  a. Gradient Boosting algorithm known for high performance

  b. Handles complex feature interactions

  c. Robust to overfitting

  d. Provides feature importance insights

## Model Training Strategy
ColumnTransformer for parallel preprocessing

Pipeline integration for reproducible workflows

Stratified K-Fold cross-validation

RandomizedSearchCV for hyperparameter tuning

## Model Evaluation Metrics
The project evaluates models using comprehensive metrics:

Accuracy: Overall prediction correctness

Precision: Quality of positive predictions

Recall: Ability to find all positive samples

F1-Score: Harmonic mean of precision and recall

ROC-AUC: Model discrimination capability

Confusion Matrix: Detailed error analysis

## Key Features
### Technical Strengths
Comprehensive Feature Engineering: 49 engineered features capturing diverse customer aspects

Robust Preprocessing: Automated pipelines for scalable deployment

Model Comparison: Multiple algorithms with rigorous evaluation

Production Ready: XGBoost optimized for real-world performance

Interpretability: Feature importance analysis for business insights

## Business Value
Proactive Retention: Identify at-risk customers before churn

Cost Reduction: Lower customer acquisition costs through better retention

Targeted Marketing: Focus resources on high-value retention opportunities

Customer Insights: Understand drivers of churn for service improvements


## Installation & Usage
Prerequisites
bash
Python 3.8+
pip install -r requirements.txt

# Required Libraries
python
## Core Data Handling
pandas, numpy

## Visualization
matplotlib, seaborn

## Machine Learning
scikit-learn, xgboost

## Model Persistence
joblib

## Running the Project
Clone the repository

Install dependencies: pip install -r requirements.txt

Run the Jupyter notebook: jupyter notebook CHURN_MODEL_TELCO.ipynb

Execute cells sequentially for full pipeline

## Expected Outcomes
High-performance churn prediction with actionable probabilities

Feature importance rankings to guide business strategy

Model interpretability for stakeholder trust

Scalable pipeline for ongoing model retraining


