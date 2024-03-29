# Dynamic XGBoost-based model on a data stream for stock price prediction
In this Kaggle competition [https://www.kaggle.com/competitions/optiver-trading-at-the-close] , the goal was to predict closing prices of 200 different stocks, at the last 10 minutes of the NASDAQ closing auction. The training data encompassed 480 days of auctions, while the scoring was done on the unknown test set, which was revealed day by day since the start of the competition. This means that many of the tasks had to be automatized for a data stream, including  data collection and cleaning, feature engineering, model retraining, and prediction.

# Project Overview
This project revolved around XGBoost Machine Learning Algorithm, a gradient-boosting tree.

As the final submission allowed for two strategies, two models were developed:
1) One models that encompassed all 200 stocks, with retraining after each day.
2) For each stock, a individual model, without the retraining.
   
# Technical Overview
[1. k_fold_time_series_hyperoptimization.ipynb](https://github.com/lukablagoje/dynamic-XGBoost-model-data-stream-prediction/blob/main/1.%20k_fold_time_series_hyperoptimization.ipynb) - Optimized hyperparameters using k-fold cross-validation tailored for time-series data, including periodic retraining.

[2. data_stream_prediction.ipynb](https://github.com/lukablagoje/dynamic-XGBoost-model-data-stream-prediction/blob/main/2.%20data_stream_prediction.ipynb) - Automated data-stream tasks: data collection and cleaning, feature engineering, model retraining, and prediction.

# Data 
Accessible under the rules of the competition at this link: [https://www.kaggle.com/competitions/optiver-trading-at-the-close](https://www.kaggle.com/competitions/optiver-trading-at-the-close)
