# Dynamic XGBoost-based model for real-time price prediction
In this Kaggle competition [https://www.kaggle.com/competitions/optiver-trading-at-the-close] , the goal was to predict closing prices of 200 different stocks, at the last 10 minutes of the NASDAQ closing auction. The training data encompassed 480 days of auctions, while the scoring was done on the unknown test set, unavailable to the public.

# Project Overview
This project revolved around XGBoost Machine Learning Algorithm, a gradient-boosting tree.

As the final submission allowed for two strategies, two models were developed:
1) One models that encompassed all 200 stocks, with retraining after each day.
2) For each stock, a individual model, without the retraining.
   
# Technical Overview

1) Initial exploratory data analysis - data_analysis_luka.ipynb
2) Cross-validation framework for time series data - https://github.com/lukablagoje/closing-cross-auction/blob/main/hyperparameter_optimization_individual_models.py
3) Overall exploration of the cross-validation - https://github.com/lukablagoje/closing-cross-auction/blob/main/hyperparameter_optimization_results.ipynb
4) Checking how model features were change during retraining - [model_diagnostic_evaluation_framework.ipynb
](https://github.com/lukablagoje/closing-cross-auction/blob/main/model_diagnostic_evaluation_framework.ipynb)https://github.com/lukablagoje/closing-cross-auction/blob/main/model_diagnostic_evaluation_framework.ipynb
