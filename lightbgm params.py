import lightgbm as lgb
from scipy.stats import randint as sp_randint

param = {
    "num_leaves": sp_randint(1, 20) # int, higher = higher accuracy but more overfitting
    "min_data_in_leaf": 
    "max_depth": # int, -1 for no limit
    "bagging_fraction":
    "feature_fraction":
    "max_bin": #int, default = 255
    
}
