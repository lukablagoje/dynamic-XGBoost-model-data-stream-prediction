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
print("Loading data")
dtypes = {
    'stock_id' : np.uint8,
    'date_id' : np.uint16,
    'seconds_in_bucket' : np.uint16,
    'imbalance_buy_sell_flag' : np.int8,
    'time_id' : np.uint16,
}
df = pd.read_csv('train.csv', dtype = dtypes).drop(['row_id','time_id'], axis = 1)

df['far_price'] = df.apply(
    lambda row: -1 if np.isnan(row['far_price']) else row['far_price'],
    axis=1)

df['near_price'] = df.apply(
    lambda row: -1 if np.isnan(row['near_price']) else row['near_price'],
    axis=1)
df = df.dropna()
X_entire_dataset, y_entire_dataset = df.iloc[:,:-1].copy(),df.iloc[:,-1].copy()
x_columns = X_entire_dataset.columns

print("Creating folds")
number_of_stocks = 200
number_of_bucket_iter = 55

# Creating time series folds
fold_dict = {}
X_train_dict = {}
y_train_dict = {}
X_test_dict = {}
y_test_dict = {}
number_of_folds = 5
total_number_of_days = 480
fold_size =  total_number_of_days/number_of_folds
rolling_test_size = fold_size//10
for f_i in range(1,number_of_folds+1):

    fold = df.loc[(df['date_id'] <= f_i * fold_size)& (df['date_id']  > (f_i-1) * fold_size)]
    X_train = df.loc[(df['date_id'] < f_i * fold_size - rolling_test_size)& (df['date_id']  > (f_i-1) * fold_size)][x_columns]
    y_train = df.loc[(df['date_id'] < f_i * fold_size - rolling_test_size)& (df['date_id']  > (f_i-1) * fold_size)][['target']]
    
    X_test =  df.loc[(df['date_id'] <= f_i * fold_size)& (df['date_id']  >= f_i * fold_size - rolling_test_size)][x_columns]
    y_test = df.loc[(df['date_id'] <= f_i * fold_size)& (df['date_id']   >= f_i * fold_size - rolling_test_size)][['target']]
    fold_dict[f_i] = fold.copy()
    
    X_train_dict[f_i] =  X_train.copy()
    y_train_dict[f_i] =  y_train.copy()
    X_test_dict[f_i] =  X_test.copy()
    y_test_dict[f_i] =  y_test.copy()
    
#Hyperparameter testing
max_depth_list = [3,5,10,15,20]
gamma_list = [1,5,10]
min_child_list = [0,2,5,10]
n_estimators_list = [100,200,300]
subsample_list = [0.1,0.5,0.75,1]
learning_rate_list = [0.01,0.05,0.1,0.2]
total_iter = len(max_depth_list) * len(gamma_list) * len(min_child_list) * len(n_estimators_list) * len(learning_rate_list) * len(subsample_list)
result_mean_dict = {}
result_std_dict = {}
results_df_dict = {'max_depth':[],'gamma':[],'n_estimator':[],'min_child':[],'learning_rate':[],'mean_mae':[],'subsample':[],'std_mae':[],'daily_var_mean':[]}
count = 1
starting_count = 0 # len(result_df)
if starting_count == 0:
    results_df_dict = {'max_depth':[],'gamma':[],'n_estimator':[],'min_child':[],'learning_rate':[],'mean_mae':[],'subsample':[],'std_mae':[],'daily_var_mean':[]}
else:
    results_df_dict = {}
    for hyperparameter in  result_df.to_dict():
        results_df_dict[hyperparameter] = []
        for h_value in results_df.to_dict()[key].values():
            results_df_dict [hyperparameter].append(h_value)
print('Hyperparameter evaluation starting')
for learning_rate in learning_rate_list:
    for min_child in min_child_list:
        for subsample in subsample_list:
            for n_estimator in n_estimators_list:
                for gamma in gamma_list:
                    for max_depth in max_depth_list:
                        # This if-statement is used to continue from the uploaded dataframe of reuslts
                        start = time.time()
                        if count > starting_count:
                            result_list = []
                            result_variance_mean_list = []
                            for f_i in range(1,number_of_folds+1):
                                # Model is pretrained with the training data
                                model = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                                           n_estimators=n_estimator,
                                           objective='reg:squarederror',
                                           max_depth=max_depth,
                                           learning_rate=learning_rate,
                                            min_child_weight=min_child,
                                            subsample = subsample,
                                            gamma = gamma)

                                model.fit(X_train_dict[f_i] , y_train_dict[f_i])
                                booster = model.get_booster()
                                daily_result_list = []
                                for date_id in X_test_dict[f_i]['date_id'].unique():
                                    print('Validation',count,'out of',total_iter,'| Fold:',f_i,'out of',number_of_folds,'| Date:',date_id,'from',min( X_test_dict[f_i]['date_id'].unique()),'to', max(X_test_dict[f_i]['date_id'].unique()))
                                    # Retraining the model with previous day data
                                    if date_id > min(X_test_dict[f_i]['date_id'].unique()):
                                        X_previous_day_test = X_test_dict[f_i][X_test_dict[f_i]['date_id'] == date_id - 1].copy()
                                        y_previous_day_test =  y_test_dict[f_i][X_test_dict[f_i]['date_id'] == date_id - 1].copy()
                                        model.fit(X_previous_day_test , y_previous_day_test,xgb_model=booster)
                                    #Predicting the current day data
                                    X_current_test = X_test_dict[f_i][X_test_dict[f_i]['date_id'] == date_id].copy()
                                    y_current_test = y_test_dict[f_i][X_test_dict[f_i]['date_id'] == date_id].copy()
                                    y_predict = model.predict(X_current_test[X_current_test['date_id'] == date_id])
                                    result_daily = mean_absolute_error(y_predict , y_current_test)
                                    daily_result_list.append(result_daily)

                                result_fold = np.mean( daily_result_list)
                                result_list.append(result_fold)
                                result_daily_variance = np.std( daily_result_list)/np.mean( daily_result_list)
                                result_variance_mean_list.append(result_daily_variance)

                            result_mean = np.mean(result_list)
                            result_std = np.std(result_list)
                            result_variance_mean = np.mean(result_variance_list)

                            results_df_dict['daily_var_mean'].append( result_variance_mean)
                            results_df_dict['mean_mae'].append(result_mean)
                            results_df_dict['std_mae'].append(result_std)
                            results_df_dict['max_depth'].append(max_depth)
                            results_df_dict['gamma'].append(gamma)
                            results_df_dict['n_estimator'].append(n_estimator)
                            results_df_dict['min_child'].append(min_child)
                            results_df_dict['learning_rate'].append(learning_rate)
                            results_df_dict['subsample'].append(subsample)
                            result_df = pd.DataFrame(results_df_dict)
                            result_df.to_csv('eval_xgboost_hyperameter_results_n_folds_' + str(number_of_folds) + '.csv')
                            end = time.time()
                            print('**** Validation',count,'out of',total_iter,'Total time',np.round(end - start,2),'****')
                            print('MAE:',result_mean,'+-',result_std)
                            print('max_depth:', max_depth,'gamma:',gamma,'n_estimator:',n_estimator,'min_child:',min_child,'learning_rate:',learning_rate)
                        count+=1
