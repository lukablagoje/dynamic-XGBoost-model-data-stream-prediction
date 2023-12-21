import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import time
import matplotlib.pyplot as plt
def create_folds(number_of_folds,rolling_test_size_ratio,fold_size,df):
    rolling_test_size = int(fold_size*rolling_test_size_ratio)
    print("Creating",number_of_folds,'folds')
    fold_dict = {}
    X_train_dict = {}
    y_train_dict = {}
    X_test_dict = {}
    y_test_dict = {}
    for f_i in range(1,number_of_folds+1):
        train_start_date = (f_i-1) * fold_size
        train_end_date = f_i * fold_size - rolling_test_size
        test_start_date = f_i * fold_size - rolling_test_size
        test_end_date =  f_i * fold_size
        
        fold = df.loc[(df['date_id'] < test_end_date)& (df['date_id']  >= train_start_date)]
        X_train = df.loc[(df['date_id'] < train_end_date)& (df['date_id']  >= train_start_date)][X_columns]
        y_train = df.loc[(df['date_id'] < train_end_date)& (df['date_id']  >= train_start_date)][['target']]
        
        X_test =  df.loc[(df['date_id'] < test_end_date)& (df['date_id']  >=  test_start_date )][X_columns]
        y_test = df.loc[(df['date_id'] < test_end_date)& (df['date_id']   >=  test_start_date )][['target']]
        if f_i == number_of_folds:
            fold = df.loc[(df['date_id'] <= test_end_date)& (df['date_id']  >= train_start_date)]
            X_test =  df.loc[(df['date_id'] <= test_end_date)& (df['date_id']  >=  test_start_date )][X_columns]
            y_test = df.loc[(df['date_id'] <= test_end_date)& (df['date_id']   >=  test_start_date )][['target']]
        fold_dict[f_i] = fold.copy()
        
        X_train_dict[f_i] =  X_train.copy()
        y_train_dict[f_i] =  y_train.copy()
        
        X_test_dict[f_i] =  X_test.copy()
        y_test_dict[f_i] =  y_test.copy()
    return fold_dict.copy(),X_train_dict.copy(), y_train_dict.copy(),X_test_dict.copy(), y_test_dict.copy()

def evaluate_folds(df,fold_dict,X_train_dict, y_train_dict,X_test_dict, y_test_dict,par_dict):
            mae_test_list = []
            mae_train_list = []
            for f_i in range(1,number_of_folds+1):
                start_fold = time.time()
                # Model is pretrained with the training data
                model = xgb.XGBRegressor(base_score=0, booster='gbtree',    
                           n_estimators=par_dict['n_estimators'],
                           objective='reg:squarederror',
                           max_depth=par_dict['max_depth'],
                           eta=par_dict['eta'],
                            min_child_weight=par_dict['min_child'],
                            subsample =par_dict['subsample'],
                            gamma =par_dict['gamma'],
                            reg_lambda = par_dict['lambda'])
                X_seconds_list = []
                
                # Setting up the dataframes
                X_train_original = X_train_dict[f_i].copy()
                y_train_original = y_train_dict[f_i].copy()
                X_current_dataset = X_train_dict[f_i].copy()
                y_current_dataset = y_train_dict[f_i].copy()
                
                # Training the model and giving predictions on the test set
                model.fit(X_train_original , y_train_original)
                y_train_prediction_results = model.predict(X_train_original)
                X_train_original['target_pred'] = list(y_train_prediction_results)
                X_train_original['target']  = y_train_original
                X_train_original['absolute_error'] = abs(X_train_original['target'] - X_train_original['target_pred'])
                mae_train_list.append(np.round(X_train_original['absolute_error'].mean(),5))
                
                for date_id in sorted(X_test_dict[f_i]['date_id'].unique()):
                    # Retraining the model with previous day data
                    test_start_date_id =  min(X_test_dict[f_i]['date_id'].unique())
                    if date_id > test_start_date_id:
                        X_previous_day_test = X_test_dict[f_i][X_test_dict[f_i]['date_id'] == date_id - 1].copy()
                        y_previous_day_test =  y_test_dict[f_i][X_test_dict[f_i]['date_id'] == date_id - 1].copy()
                        
                        # Retraining the model
                        X_current_dataset = pd.concat([X_train_dict[f_i],X_previous_day_test ]).reset_index(drop=True).copy()
                        y_current_dataset = pd.concat([y_train_dict[f_i],y_previous_day_test]).reset_index(drop=True).copy()
                        model.fit(X_current_dataset , y_current_dataset, xgb_model =  model.get_booster())    
                        
                    #Predicting the current day data
                    X_day_test = X_test_dict[f_i][X_test_dict[f_i]['date_id'] == date_id].copy()
                    y_day_test = y_test_dict[f_i][X_test_dict[f_i]['date_id'] == date_id].copy()
                    
                    # Predicting for each batch of 10 seconds
                    for seconds_in_bucket in sorted(list(X_day_test['seconds_in_bucket'].unique())):
                        X_seconds_test = X_day_test[X_day_test['seconds_in_bucket'] == seconds_in_bucket].copy()
                        y_seconds_test = y_day_test[X_day_test['seconds_in_bucket'] == seconds_in_bucket].copy()
  
                        # Testing predictions
                        X_seconds_test['target_pred'] = list(model.predict(X_seconds_test))
                        X_seconds_test['target'] = y_seconds_test.copy()
                        X_seconds_list.append(X_seconds_test.copy())
                
                    end_fold = time.time()
                    print('Date:',date_id,'from',min( X_test_dict[f_i]['date_id'].unique()),'to', max(X_test_dict[f_i]['date_id'].unique()),'| Total time spent on this fold',np.round((end_fold - start_fold)/60,2),'minutes ****')                
                X_test_df = pd.concat(X_seconds_list).copy()
                X_test_df['absolute_error'] = abs(X_test_df['target'] - X_test_df['target_pred'])
                mae_test_list.append(np.round(X_test_df['absolute_error'].mean(),5))
            return  mae_test_list.copy(), mae_train_list.copy()

print("Loading data")
dtypes = {
    'stock_id' : np.uint8,
    'date_id' : np.uint16,
    'seconds_in_bucket' : np.uint16,
    'imbalance_buy_sell_flag' : np.int8,
    'time_id' : np.uint16,
}

X_columns = ['stock_id', 'date_id', 'seconds_in_bucket', 'imbalance_size',
       'imbalance_buy_sell_flag', 'reference_price', 'matched_size',
       'far_price', 'near_price', 'bid_price', 'bid_size', 'ask_price',
       'ask_size', 'wap','first_change','second_change','time_id']
y_columns = ['target']
df = pd.read_csv('train.csv', dtype = dtypes).drop(['row_id'], axis = 1)

df['far_price'] = df.apply(
    lambda row: -1 if np.isnan(row['far_price']) else row['far_price'],
    axis=1)

df['near_price'] = df.apply(
    lambda row: -1 if np.isnan(row['near_price']) else row['near_price'],
    axis=1)

df['first_change'] = df.apply(lambda row: -1 if row['seconds_in_bucket'] <= 290 else 1,axis=1)

df['second_change'] = df.apply(lambda row: -1 if row['seconds_in_bucket'] <= 480 else 1,axis=1)


df = df.dropna()

print("Creating folds")
number_of_stocks = 200
number_of_bucket_iter = 55
# Creating time series folds
number_of_folds = 3
total_number_of_days = 480
fold_size =  total_number_of_days/number_of_folds
rolling_test_size_ratio = 0.1
max_depth_list = [3,5,17]
n_estimators_list = [10,50,1000]
gamma_list = [0,10,100]
lambda_list = [0,10,100]
min_child_list = [0,10,100]
subsample_list = [1,0.75,0.5]
eta_list = [0.9,0.3,0.03]
total_iter = len(max_depth_list) * len(n_estimators_list) * len(gamma_list) * len(min_child_list)*len(subsample_list)*len(eta_list)*len(lambda_list)
result_mean_dict = {}
result_std_dict = {}
results_df_dict =  {'max_depth':[],'n_estimators':[],'eta':[],'min_child':[],'subsample':[],'gamma':[],'lambda':[],'train_mean_mae':[],'train_std_mae':[],'test_mean_mae':[],'test_std_mae':[]}
count = 1

print('Hyperparameter evaluation starting')
method_dict = {'model_name' : 'all_stocks_xgboost',
'retraining_freq' : 'daily_retraining',
'retraining_method' : 'on_full_data'}
for subsample in subsample_list:
    for min_child in min_child_list:
        for gamma in gamma_list:
            for lambda_p in lambda_list:
                for eta in eta_list:
                    for n_estimators in n_estimators_list:
                        for max_depth in max_depth_list:
                            start = time.time()
                            par_dict = {}
                            par_dict['n_estimators'] = n_estimators
                            par_dict['max_depth'] = max_depth
                            par_dict['gamma'] = gamma
                            par_dict['subsample'] = subsample
                            par_dict['min_child'] = min_child
                            par_dict['eta'] = eta
                            par_dict['lambda'] = lambda_p

                            results_df_dict['n_estimators'].append(n_estimators)        
                            results_df_dict['max_depth'].append(max_depth)
                            results_df_dict['gamma'].append(gamma)
                            results_df_dict['subsample'].append(subsample)
                            results_df_dict['min_child'].append(min_child)
                            results_df_dict['eta'].append(eta) 
                            results_df_dict['lambda'].append(lambda_p) 

                            fold_dict,X_train_dict, y_train_dict,X_test_dict, y_test_dict = create_folds(number_of_folds,rolling_test_size_ratio,fold_size,df.copy())
                            mae_test_list, mae_train_list = evaluate_folds(df.copy(),fold_dict.copy(),X_train_dict.copy(), y_train_dict.copy(),X_test_dict.copy(), y_test_dict.copy(),par_dict.copy())

                            test_mean = np.round(np.mean(mae_test_list.copy()),5)
                            test_std = np.round(np.std(mae_test_list.copy()),5)
                            results_df_dict['test_mean_mae'].append(test_mean)
                            results_df_dict['test_std_mae'].append(test_std)

                            train_mean = np.round(np.mean(mae_train_list.copy()),5)
                            train_std = np.round(np.std(mae_train_list.copy()),5)        
                            results_df_dict['train_mean_mae'].append(train_mean)
                            results_df_dict['train_std_mae'].append(train_std)

                            result_df = pd.DataFrame(results_df_dict)
                            result_df.to_csv(method_dict['model_name'] + '_' + method_dict['retraining_freq'] + '_' + method_dict['retraining_method']+ '_n_folds_' + str(number_of_folds) + '.csv')
                            end = time.time()
                            print('**** Validation',count,'out of',total_iter,'Total time spent on these hyperparameters',np.round((end - start)/60,2),'minutes ****')
                            count+=1
