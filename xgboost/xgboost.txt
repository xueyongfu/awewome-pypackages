import xgboost as xgb
import numpy as np

data = np.random.rand(5, 10)  # 5 entities, each contains 10 features
label = np.random.randint(2, size=5)  # binary target
dtrain = xgb.DMatrix(data, label=label)

data = np.random.rand(5, 10)  # 5 entities, each contains 10 features
label = np.random.randint(2, size=5)  # binary target
dtest = xgb.DMatrix(data, label=label)

param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'

param['eval_metric'] = ['auc', 'ams@0']


evallist = [(dtest, 'eval'), (dtrain, 'train')]

# train
num_round = 10
bst = xgb.train(param, dtrain, num_round, evallist)

# save
bst.save_model('xgb.model')

# The model and its feature map can also be dumped to a text file.

# dump model
bst.dump_model('dump.raw.txt')
# dump model with feature map
# bst.dump_model('dump.raw.txt', 'featmap.txt')

# A saved model can be loaded as follows:

bst = xgb.Booster({'nthread': 4})  # init model
bst.load_model('xgb.model')  # load data








