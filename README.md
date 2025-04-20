# Regularized Regression Models

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0-green.svg)](https://scikit-learn.org/stable/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3-yellow.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21-orange.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4-red.svg)](https://matplotlib.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository implements and compares three regularized regression techniques: Ridge Regression, Lasso Regression, and Elastic Net Regression. The project demonstrates how these advanced regression methods can be used to handle high-dimensional data, reduce overfitting, and improve prediction accuracy. Using baseball players' statistics (Hitters dataset), the models predict player salaries while handling multicollinearity and feature selection.

## Table of Contents

- [Regularized Regression Models](#regularized-regression-models)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Dataset](#dataset)
  - [Models Implemented](#models-implemented)
    - [1. Ridge Regression](#1-ridge-regression)
    - [2. Lasso Regression](#2-lasso-regression)
    - [3. Elastic Net Regression](#3-elastic-net-regression)
  - [Features](#features)
  - [Methodology](#methodology)
  - [Results](#results)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Dependencies](#dependencies)

## Introduction

Regularized regression techniques add penalty terms to the ordinary least squares objective function to reduce model complexity and prevent overfitting. This project explores three popular regularization methods:

1. **Ridge Regression**: Uses L2 regularization to shrink coefficients, handling multicollinearity effectively
2. **Lasso Regression**: Uses L1 regularization that can perform feature selection by forcing some coefficients to zero
3. **Elastic Net Regression**: Combines L1 and L2 regularization, providing the benefits of both approaches

The repository demonstrates the implementation, tuning, and comparison of these models for predictive accuracy.

## Dataset

The analysis uses the "Hitters" dataset, which contains baseball player statistics and their salaries. 

Key variables include:
- Player performance metrics (AtBat, Hits, HmRun, Runs, RBI, Walks, etc.)
- Career statistics (CAtBat, CHits, CHmRun, CRuns, etc.)
- Player attributes (Years of experience, League, Division, etc.)
- Target variable: Salary (in thousands of dollars)

## Models Implemented

### 1. Ridge Regression
- Adds L2 penalty (sum of squared coefficients) to the cost function
- Shrinks coefficients toward zero but does not eliminate variables
- Well-suited for data with multicollinearity issues

### 2. Lasso Regression
- Adds L1 penalty (sum of absolute coefficients) to the cost function
- Can perform automatic feature selection by reducing some coefficients to exactly zero
- Useful for high-dimensional data with many irrelevant features

### 3. Elastic Net Regression
- Combines L1 and L2 penalties
- Balances the feature selection capability of Lasso with the coefficient shrinkage of Ridge
- Particularly effective when features are correlated

## Features

- Implementation of Ridge, Lasso, and Elastic Net regression models
- Visualization of coefficient paths as regularization strength changes
- Cross-validation for automatic penalty parameter (alpha) selection
- Model tuning to optimize performance
- Comprehensive model evaluation and comparison
- Feature importance analysis

## Methodology

1. **Data Preprocessing**:
   - Handling missing values
   - One-hot encoding of categorical variables
   - Train-test splitting

2. **Model Development**:
   - Implementation of base Ridge, Lasso, and Elastic Net models
   - Visualization of regularization paths for different alpha values
   - Cross-validation to find optimal regularization strength

3. **Model Tuning**:
   - Implementing RidgeCV, LassoCV, and ElasticNetCV
   - Finding optimal alpha values for each model
   - Training models with optimal parameters

4. **Evaluation**:
   - Calculating Root Mean Squared Error (RMSE)
   - Computing R² scores
   - Comparing model performance before and after tuning

## Results

The analysis revealed the following key insights:

- **Performance Comparison**:
  - Lasso Regression performed slightly better than Ridge and Elastic Net before tuning
  - After tuning, Ridge and Lasso showed similar performance (R² ≈ 0.355)
  - Elastic Net had the lowest performance (R² ≈ 0.282)

- **RMSE Scores**:
  - Ridge: 357.05 (untuned), 373.60 (tuned)
  - Lasso: 356.10 (untuned), 373.60 (tuned)
  - Elastic Net: 357.17 (untuned), 394.15 (tuned)

- **Feature Selection**:
  - Lasso eliminated several features, focusing on Hits, Walks, CHits, CRuns, CRBI, CWalks, and PutOuts
  - Ridge kept all features but with shrunk coefficients
  - Elastic Net was the most selective, retaining only a few features

Based on the results, either Ridge or Lasso regression would be recommended for this dataset, with Lasso potentially preferred if feature selection is desired.

## Installation

1. Clone this repository
```bash
git clone https://github.com/cnosmn/Ridge-Lasso-and-ElasticNet-Regression-Models-Prediction.git
cd Ridge-Lasso-and-ElasticNet-Regression-Models-Prediction
```

2. Install required packages
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Usage

1. Run the Jupyter Notebook or Python script to perform the analysis
```bash
jupyter notebook model_predict.ipynb
```

2. Modify the hyperparameters or try different datasets to explore model behavior
```python
# Example: Training a Ridge model with custom alpha
ridge_model = Ridge(alpha=10.0).fit(X_train, y_train)
```

## Dependencies

- Python 3.x
- NumPy: For numerical operations
- Pandas: For data manipulation and analysis
- scikit-learn: For implementing machine learning models
- Matplotlib: For data visualization
- Seaborn: For enhanced visualizations

---

Feel free to contribute to this repository by adding more regularization techniques or improving the existing implementations. If you have any questions or suggestions, please open an issue or submit a pull request.