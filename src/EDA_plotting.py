import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def pairplotting(data, features_list, title):
    plt.figure(figsize=(12,12))
    sns.pairplot(data[features_list], diag_kind = 'kde')
    plt.xticks(rotation= 85, fontsize= 10)
    # plt.tight_layout(pad= 3)
    plt.title(title, fontsize= 30)
    plt.show();


if __name__ == '__main__':
    
potential_features = X_train.columns.tolist()

# list of features reduced by initial EDA
pairplotting(X_train, potential_features, "All potential Features")


drug_related_features = ['mme_percap','drugdeathrate','partD30dayrxrate','pctunins','num_SSPs','bup_phys','drugdep','pctunmetneed','nonmedpain']
income_related_features = ['unemployment_rate','poverty_rate','household_income','Mean income','Median income',]
msm_features = ['ADULTMEN','MSM12MTH','MSM5YEAR','%msm12month','%msm5yr']
facilities_features = ['MH_fac','Med_MH_fac','Num_drug_treatment_fac','Num_med_drug_treatment_fac']

# Drug related features pairplot
pairplotting(X_train, drug_related_features, 'Drug Related Features')

# Income related features pairplot
pairplotting(X_train, income_related_features, 'Income Related Features')

# MSM features
pairplotting(X_train, msm_features, 'MSM Features')

# Facilities features
pairplotting(X_train, facilities_features, 'Facilities Features')

# the reduced list of potential features from EDA and visualization:
reduced_feature_list = ['Population','drugdeathrate','mme_percap','partD30dayrxrate','pctunins','num_SSPs','pctunmetneed',
                        '%msm12month','MSM_recent_ratio','poverty_rate','Mean income','Num_drug_treatment_fac','Percent_men','income_ratio']

# Reduced features
pairplotting(X_train, reduced_feature_list, 'Reduced Features')


