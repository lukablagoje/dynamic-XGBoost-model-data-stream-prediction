# Dynamic XGBoost-based model on a data stream for stock price prediction
In this Kaggle [competition](https://www.kaggle.com/competitions/optiver-trading-at-the-close), the goal was to predict the closing prices of 200 different stocks, at the last 10 minutes of the NASDAQ closing auction. The training data encompassed 480 days of auctions, while the scoring was done on the unknown test set, which was revealed day by day since the start of the competition. This means that many of the tasks had to be automatized for a data stream, including  data collection and cleaning, feature engineering, model retraining, and prediction. To predict the prices, a machine learning model was used, specifically XGBoost Regressor (displayed in the figure). As the final submission allowed for two strategies, there were two variants of the final prediction system:
1) One model that encompassed all 200 stocks, with retraining after each day.
2) For each stock, an individual model, without the retraining.
   

![image](https://github.com/lukablagoje/dynamic-XGBoost-model-data-stream-prediction/assets/52599010/3caa45c6-de51-41ac-908f-430b9b66443b)
# Technical Overview
The project is split into two parts:

[1. k_fold_time_series_hyperoptimization.ipynb](https://github.com/lukablagoje/dynamic-XGBoost-model-data-stream-prediction/blob/main/1.%20k_fold_time_series_hyperoptimization.ipynb) - Found optimal hyperparameters using k-fold cross-validation tailored for time-series data, including periodic retraining.

[2. data_stream_prediction.ipynb](https://github.com/lukablagoje/dynamic-XGBoost-model-data-stream-prediction/blob/main/2.%20data_stream_prediction.ipynb) - Automated data-stream tasks: data collection and cleaning, feature engineering, model retraining, and prediction for the final submission in the competition.

# Data 
Accessible under the rules of the competition at this link: [https://www.kaggle.com/competitions/optiver-trading-at-the-close](https://www.kaggle.com/competitions/optiver-trading-at-the-close)

Kaggle Notebook: https://www.kaggle.com/code/lukablagoje/xgboost-pipeline

Image Source: https://www.researchgate.net/figure/Simplified-structure-of-XGBoost_fig2_348025909
