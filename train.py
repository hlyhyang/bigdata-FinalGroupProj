import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import metrics
from IPython import embed
from tqdm import tqdm
import datetime

print('Loading data...')
train_data = pd.read_csv('~/kkbox/data/fe_train.csv')
test_data = pd.read_csv('~/kkbox/data/fe_train.csv')
print('Loading data finished')



def train(train_data, test_data, num_rounds, learn_rate, boosting, device):

    for col in train_data.columns:
        if train_data[col].dtype == object:
            train_data[col] = train_data[col].astype('category')
            test_data[col] = test_data[col].astype('category')

    # divided validation
    train_val = train_data.tail(int(len(train_data) * 0.2))
    train_trn = train_data.head(len(train_data) - int(len(train_data) * 0.2))

    params_gdbt = {
        'objective': 'binary',
        'boosting': boosting,
        'learning_rate': learn_rate,
        'verbose': 5,
        'num_leaves': 128,
        'bagging_fraction': 0.9,
        'bagging_freq': 1,
        'bagging_seed': 2017,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 2017,
        'max_bin': 512,
        'max_depth': 16,
        'num_rounds': num_rounds,
        'metric': 'auc',
        'device': device
    }

    d_train = lgb.Dataset(train_trn.drop('target', axis=1), train_trn['target'].values)
    watchlist = lgb.Dataset(train_val.drop('target', axis=1), train_val['target'].values)

    # model training
    print('start model training')
    model_gdbt_local = lgb.train(params_gdbt, train_set=d_train, valid_sets=watchlist, verbose_eval=5,
                                 early_stopping_rounds=5)  
    print('model training finished')

    local_result = model_gdbt_local.predict(train_val.drop('target', axis=1))  
    local_score = metrics.roc_auc_score(train_val['target'], local_result)  
    print(local_score)

    theTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    model_gdbt_local.save_model('/home/paperspace/kkbox/result/{}.model'.format(theTime))
    f = open('~/kkbox/result/logs.log', 'a+')
    f.write('Update Time: {}\n'.format(theTime))
    f.write('Model Parameters: {}\n'.format(params_gdbt))
    f.write('Model AUC: {}\n'.format(local_score))
    f.write('Model Path: {}\n'.format('~/kkbox/result/{}.model'.format(theTime)))
    f.write('\n\n')
    f.close()

    return local_score, model_gdbt_local


online_result_gbdt, model_online_gbdt = train(train_data=train_data, test_data=test_data, num_rounds=3000, learn_rate=0.05,
                                                    boosting='gbdt', device='cpu')

