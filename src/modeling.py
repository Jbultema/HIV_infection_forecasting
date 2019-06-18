import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# this function will iteratively model all the features, test removing the worst feature, and see if it improves model fit

import statsmodels.formula.api as sm
from sklearn.preprocessing import StandardScaler

def backwardElimination(X, Y, SL):
    numVars = X.shape[1]
    temp = np.zeros((X.shape)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, X).fit()
        maxVar = max(regressor_OLS.pvalues)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = X[:, j]
                    X = np.delete(X, j, 1)
                    tmp_regressor = sm.OLS(Y, X).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        X_rollback = np.hstack((X, temp[:,[0,j]]))
                        X_rollback = np.delete(X_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return X_rollback
                    else:
                        continue
    print(regressor_OLS.summary())
    return X

# use this elastic_slide function to use Lasso, Ridge, or a combination of the two regression models

def elastic_slide(X_data, y_data, L1_value= 1, alpha_range= 1, best_model= False):
    import statsmodels.api as sm
    model = sm.OLS(y, X)
    results_1 = model.fit()
    final_results = np.array([]).reshape(-1,4)
    params = []

    for j in alpha_range:
        results_2 = model.fit_regularized(L1_wt=L1_value, alpha=j, start_params=results_1.params)
        final = sm.regression.linear_model.OLSResults(model, results_2.params, model.normalized_cov_params)
        score= [L1_value, j, final.rsquared_adj, final.condition_number]
        final_results = np.vstack([final_results, score])
        params.append([x for x in final.params])

    if best_model:
        return final , final_results , np.array(params) , params[np.argmax(final_results[:,2], out=None)]
    return final , final_results , np.array(params)


if __name__ == '__main__':

    # lets test this with the full set of features selected from EDA
    backward_elim_results = backwardElimination(X_train[reduced_feature_list].values,y_train.values, 0.05)

    # run again with same hyper-params and see if we get the same features
    backward_elim_results2 = backwardElimination(X_train[reduced_feature_list].values,y_train.values, 0.05)

    # testing the stepwise backward elimination
    backward_elim_results3 = backwardElimination(X2_train[reduced_feature_list].values,np.log(y2_train.values), 0.05)
    results3_df = pd.DataFrame(backward_elim_results3)

    # if we give it all possible features, what will we get back?
    X3 = df_pos_inf[potential_features]
    y3 = df_pos_inf['HIV_rel_infection']
    X3_train, X3_test, y3_train, y3_test = train_test_split(X2, y2)

    backward_elim_results4 = backwardElimination(X3_train.values,np.log(y3_train.values), 0.05)

    results4_df = pd.DataFrame(backward_elim_results4)

    EDA_selected_features =  ['Population', 'drugdeathrate', 'mme_percap', 'partD30dayrxrate', 'num_SSPs', 'pctunmetneed','%msm12month', 
                              'MSM_recent_ratio', 'poverty_rate', 'Mean income', 'Num_drug_treatment_fac', 'Percent_men']
    step_selected_features = ['Population', 'drugdeathrate', 'mme_percap', 'partD30dayrxrate', 'pctunins','num_SSPs','MSM_recent_ratio',
                              'Mean income', 'Num_drug_treatment_fac', 'Percent_men', 'pctunmetneed']

    # iteratively prune features until all EDA_selected_features are below 0.05
    EDA_selected_pruned = ['drugdeathrate', 'partD30dayrxrate', 'num_SSPs', 'pctunmetneed', '%msm12month', 'Mean income']

    # iteratively prune features until all step_selected_features are below 0.05
    step_selected_pruned = ['drugdeathrate', 'partD30dayrxrate', 'num_SSPs','pctunins', 'MSM_recent_ratio', 'Mean income']

    # these two lists converge on essentially the same features, with adjusted_R_squared of each at 0.94

    # Fitting the Linear regression model with just the features from backward_elim_results3 with good p-values
    smaller_model_EDA = sm.OLS(np.log(y2_train.values), X2_train[EDA_selected_features].values).fit()
    
    # EDA features pruned
    smaller_model_EDA_pruned = sm.OLS(np.log(y2_train.values), X2_train[EDA_selected_pruned].values).fit()

    # Fitting the Linear regression model with just the features from backward_elim_results3 (stepwise backward selected features) with good p-values
    smaller_model_step = sm.OLS(np.log(y2_train.values), X2_train[step_selected_features].values).fit()

    # step-selected features pruned
    smaller_model_step_pruned = sm.OLS(np.log(y2_train.values), X2_train[step_selected_pruned].values).fit()

    # what is the scale of our target value (HIV_rel_infection)?
    print("Test mean: ", round(y2_test.mean(),2))

    # how well does the EDA_selected_pruned model perform?
    y_pred_EDA = smaller_model_EDA_pruned.predict(X2_test[EDA_selected_pruned].values)
    EDA_RMSE = mean_squared_error(y2_test, np.exp(y_pred_EDA))
    print("EDA-selected pruned features model RMSE: ", round(EDA_RMSE, 2))

    # how well does the step_selected_pruned model perform?
    y_pred_step = smaller_model_step_pruned.predict(X2_test[step_selected_pruned].values)
    step_RMSE = mean_squared_error(y2_test, np.exp(y_pred_step))
    print("Stepwise-selected pruned features model RMSE: ", round(step_RMSE, 2))

    # run the backward stepwise feature elimination with all features to check performance
    norm_step_test = backwardElimination(X2_norm_train.values,y2_norm_train.values, 0.05)

    ## Stepward elimination on normalized dataframe: (the features may change upon each iteration)

    norm_step_features = ['Norm_partD30dayrxrate','Norm_pctunins', 'Norm_num_SSPs', 'Norm_pctunmetneed', 
                          'Norm_%msm12month', 'Norm_MSM_recent_ratio', 'Norm_Num_drug_treatment_fac', 'Norm_income_ratio']
    norm_step_pruned = ['Norm_partD30dayrxrate','Norm_pctunins', 'Norm_pctunmetneed', 
                          'Norm_%msm12month', 'Norm_MSM_recent_ratio', 'Norm_Num_drug_treatment_fac', 'Norm_income_ratio']

    # reminder that the EDA_selected_features
    EDA_selected_features =['Population','drugdeathrate','mme_percap','partD30dayrxrate','num_SSPs','pctunmetneed','%msm12month',
                        'MSM_recent_ratio','poverty_rate','Mean income','Num_drug_treatment_fac','Percent_men']
    norm_EDA_selected_features = ["Norm_" + x for x in EDA_selected_features]
    norm_EDA_pruned = ['Norm_Population','Norm_drugdeathrate','Norm_partD30dayrxrate','Norm_num_SSPs','Norm_pctunmetneed',
                 'Norm_MSM_recent_ratio','Norm_poverty_rate','Norm_Num_drug_treatment_fac']

    # Prune the backward step selected ones to below 0.05
    norm_step_test = backwardElimination(X2_norm_train[norm_step_pruned].values,y2_norm_train.values, 0.05)

    # Prune the EDA selected features on normalized data to around 0.05
    norm_EDA_test = backwardElimination(X2_norm_train[norm_EDA_pruned].values,y2_norm_train.values, 0.05)

    # how well do the normalized models perform?

    # Raw-data models
    print("Raw test mean: ", round(y2_test.mean(), 2))
    # EDA_selected_pruned model
    y_pred_EDA = smaller_model_EDA_pruned.predict(X2_test[EDA_selected_pruned].values)
    EDA_RMSE = mean_squared_error(y2_test, np.exp(y_pred_EDA))
    print("EDA-selected pruned features model RMSE: ", round(EDA_RMSE, 2))

    # step_selected_pruned model
    y_pred_step = smaller_model_step_pruned.predict(X2_test[step_selected_pruned].values)
    step_RMSE = mean_squared_error(y2_test, np.exp(y_pred_step))
    print("Stepwise-selected pruned features model RMSE: ", round(step_RMSE, 2))

    print("Normalized test mean: ", round(y2_norm.values.mean(), 2))
    # EDA normalized model
    norm_EDA_model = sm.OLS(y2_norm_train.values, X2_norm_train[norm_EDA_pruned].values).fit()
    y_pred_nEDA = norm_EDA_model.predict(X2_norm_test[norm_EDA_pruned].values)
    nEDA_RMSE = mean_squared_error(y2_norm_test.values, y_pred_nEDA)
    print("Normalized EDA-selected pruned features model RMSE: ", round(nEDA_RMSE, 3))

    # step normalized model
    norm_step_model = sm.OLS(y2_norm_train.values, X2_norm_train[norm_step_pruned].values).fit()
    y_pred_nstep = norm_step_model.predict(X2_norm_test[norm_step_pruned].values)
    nstep_RMSE = mean_squared_error(y2_norm_test.values, y_pred_nstep)
    print("Normalized step-selected pruned features model RMSE: ", round(nstep_RMSE, 3))

    # lets try an elastic net with the elastic slide function from normalized X2 values
    es_model , es_results , es_params , es_best = elastic_slide(X2_norm_train.values, y2_norm_train.values, L1_value= 1, alpha_range= [0.001, 0.01, 0.1, 0.25, 0.5, 1], best_model= True)
    es_features = [x for x in [x if abs(x[1]) > 0.01 else None for x in list(zip(X2_norm_train.columns.tolist(),es_best))] if x != None]

    # get only the names of features with coefficient values above 0.01 threshold:
    lasso_feature_names = [x[0] for x in es_features]

    # build an OLS model with only those features:
    lasso_feature_model = sm.OLS(y2_norm_train.values, X2_norm_train[lasso_feature_names].values).fit()

    y_pred_lasso = lasso_feature_model.predict(X2_norm_test[lasso_feature_names].values)
    lasso_RMSE = mean_squared_error(y2_norm_test.values, y_pred_lasso)

    print("Normalized test mean: ", round(y2_norm.values.mean(), 2))
    print("Normalized EDA-selected pruned features model RMSE: ", round(nEDA_RMSE, 3))
    print("Normalized step-selected pruned features model RMSE: ", round(nstep_RMSE, 3))
    print("Normalized lasso-selected pruned features model RMSE: ", round(lasso_RMSE, 3))