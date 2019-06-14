import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# We will be plotting lots of residuals, so lets make a function to simplify that process.

def plot_linear_residuals(model_name, model_features, training_data, save= False):
    import statsmodels.api as sm
    model_resids = model_name.outlier_test()[:,0]

    i  = int((len(model_features)/2))
    j = int((len(model_features) - i)/2)
    if len(model_features) % 2 != 0:
        i += 1
    
    fig, axs = plt.subplots(i, j, figsize= (4*i, 4*j))
    plt.tight_layout(pad=4)
    
    for idx, val in enumerate(model_features):
        plt.subplot(i,j, idx+1, label= val)
        plt.title(f'{val} Studentized Residual Plot')
        plt.scatter(training_data[val], model_resids)
        plt.hlines(0,
                      training_data[val].min(), 
                      training_data[val].max(), 
                      'k', linestyle='dashed')
        plt.xlabel(val)
        plt.ylabel('studentized residuals');

    plt.subplot(i,j, len(model_features))
    plt.scatter(model_name.fittedvalues, model_resids)
    plt.hlines(0,
                  model_name.fittedvalues.min(), 
                  model_name.fittedvalues.max(),
                  'k', linestyle='dashed')
    plt.xlabel('predicted rel HIV infection rate')
    plt.ylabel('studentized residuals')
    if save:
        plt.savefig(f'images/residuals_of_{model_name}.png', dpi= 300)
    plt.show;

    
    
    # lets see how our model predictions look for each feature selected.
# this function will make it simpler to visualize

def plot_linear_regression_features(y_pred, y_true, x_test, selected_features):
    # Visualising the Linear Regression results
    i  = int((len(selected_features)/2))
    j = int((len(selected_features) - i + 1)/2)
    fig, ax = plt.subplots(i, j, figsize= (4*i, 3*j))
    plt.ylabel('Relative HIV Infection Growth', fontsize= 20)
    plt.title('Target vs feature' , fontsize = 25)
#     plt.legend(fontsize= 15)  
    for idx, col in enumerate(selected_features):

        data1 = sorted(list(zip(x_test[col].values, y_true.values)))
        data2 = sorted(list(zip(x_test[col].values, np.exp(y_pred))))
        scat = list(zip(*data1))
        plot = list(zip(*data2))
        
        plt.subplot(i, j, idx + 1)
        sns.scatterplot(scat[0], scat[1], color = 'red', label= 'Y-test')
        sns.scatterplot(plot[0], plot[1], color = 'blue', label= "Y-predicted")
        plt.xlabel(f'{col}', fontsize= 15)

  
    plt.tight_layout(pad= 1)
    plt.show();
    
    
    
    
    
    # Plotting y_pred and y_true values for features in EDA_selected_pruned model
plot_linear_regression_features(y_pred_EDA, y2_test, X2_test, EDA_selected_pruned)

# Plotting y_pred and y_true values for features in step_selected_pruned model
plot_linear_regression_features(y_pred_step, y2_test, X2_test, step_selected_pruned)
    
    
    
    # create list of tuples of feature name and relative importance
step_weights = norm_step_model.params.tolist()
sum_step_weight = sum(step_weights)
total_step_weight = sum([abs(x) for x in step_weights])
step_rel_weights = [int(100*x/total_step_weight) for x in step_weights]
step_feature_importance = list(zip(norm_step_pruned, step_rel_weights))

EDA_weights = norm_EDA_model.params.tolist()
sum_EDA_weight = sum(EDA_weights)
total_EDA_weight = sum([abs(x) for x in EDA_weights])
EDA_rel_weights = [int(100*x/total_EDA_weight) for x in EDA_weights]
EDA_feature_importance = list(zip(norm_EDA_pruned, EDA_rel_weights))

step_results_df = pd.DataFrame(step_feature_importance, columns = ['Feature', 'Effect on prediction (%)'])
EDA_results_df = pd.DataFrame(EDA_feature_importance, columns = ['Feature', 'Effect on prediction (%)'])
step_results_df["Model"] = "Stepwise"
EDA_results_df['Model'] = "EDA"
model_results_df = pd.concat([step_results_df, EDA_results_df]).sort_values(by=['Model','Effect on prediction (%)','Feature'], ascending= [False, False, True])

model_results_df


# add these model features and impact to the results dataframe to compare all three approaches
lasso_weights = lasso_feature_model.params.tolist()
sum_lasso_weight = sum(lasso_weights)
total_lasso_weight = sum([abs(x) for x in lasso_weights])
rel_lasso_weight = [int(100 * x/total_lasso_weight) for x in lasso_weights]
lasso_feature_importance = list(zip(lasso_feature_names, rel_lasso_weight))

lasso_results_df = pd.DataFrame(lasso_feature_importance, columns = ['Feature', 'Effect on prediction (%)'])
lasso_results_df['Model'] = 'Lasso'
model_results_df2 = pd.concat([model_results_df, lasso_results_df]).sort_values(by=['Model','Feature','Effect on prediction (%)'], ascending= [True, False, True])
model_results_df2['Feature Name'] = model_results_df2['Feature'].apply(lambda x: x.split('rm_')[1])
model_results_df2[['Model', 'Feature Name', 'Effect on prediction (%)']]


# turn this into a function:
plt.figure(figsize= (15,5))
sns.catplot(x= 'Model', y= 'Effect on prediction (%)', hue= 'Feature Name', data= model_results_df2, kind= 'bar', aspect= 3, palette= 'bright', legend= False, legend_out= True)
plt.title('Impact of features on each model predictions', fontsize= 15)
plt.legend(title= "Feature Name", loc= 'right').set_bbox_to_anchor((1.2, 0.5, 0, 0))
plt.tight_layout(w_pad= 4, h_pad= 2)
plt.savefig('images/Impact_of_features.png', dpi= 300)


# lets see how our model predictions look for each feature selected.
# this function will make it simpler to visualize

def plot_linear_regression_features2(y_pred, y_true, x_test, selected_features, save= False):
    # Visualising the Linear Regression results
    i  = int((len(selected_features)/2))
    j = int((len(selected_features) - i)/2)
    if len(selected_features) % 2 != 0:
        i += 1
    fig, ax = plt.subplots(i, j, figsize= (4*i, 4*j))

    for idx, col in enumerate(selected_features):

        data1 = sorted(list(zip(x_test[col].values, y_true.values)))
        data2 = sorted(list(zip(x_test[col].values, y_pred)))
        scat = list(zip(*data1))
        plot = list(zip(*data2))
        
        plt.subplot(i, j, idx + 1)
        sns.scatterplot(scat[0], scat[1], color = 'red', label= 'Y-test')
        sns.scatterplot(plot[0], plot[1], color = 'blue', label= "Y-predicted")
        plt.xlabel(f'{col}', fontsize= 15)

  
    plt.tight_layout(pad= 0.5)
    if save:
        plt.savefig(f'images/{y_pred}_feature_true_vs_pred.png', dpi= 300)
    plt.show();
    
    
    # studentized residuals for Normalized step_pruned_model
plot_linear_residuals(norm_step_model, norm_step_pruned, X2_norm_train, True)

# Plotting y_pred and y_true values for features in Normalized EDA_pruned model
plot_linear_regression_features2(y_pred_nEDA, y2_norm_test, X2_norm_test, norm_EDA_pruned)

# studentized residuals for Normalized EDA_pruned_model
plot_linear_residuals(norm_EDA_model, norm_EDA_pruned, X2_norm_train, True)


# Plotting y_pred and y_true values for features in Normalized step_pruned model
plot_linear_regression_features2(y_pred_lasso, y2_norm_test, X2_norm_test, lasso_feature_names)


# studentized residuals for the lasso model with normalized data
plot_linear_residuals(lasso_feature_model, lasso_feature_names, X2_norm_train)

def pred_v_true_plot(true_vals, pred_val_list, labels, save= False):
    info = list(zip(labels, pred_val_list))

    fig, axs = plt.subplots(1, len(labels), figsize= (16,4), sharex= True, sharey= True, tight_layout= False)

    for idx, item in enumerate(info, 0):
        fig.add_subplot(1, len(labels), idx+1)
        sns.scatterplot(true_vals.values.reshape(-1,), item[1].reshape(-1,))
        sns.lineplot((0.1,0.9), (0.1, 0.9), color= 'k')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.ylabel("Predicted Value", fontsize= 15)
        plt.xlabel("True Value", fontsize= 15)
        plt.title(str(item[0])+' Predicted', fontsize= 20, loc= 'center')

    if save:
        plt.savefig('images/Predicted_vs_true.png', dpi= 300)
    plt.show;
    
pred_v_true_plot(y2_norm_test, [y_pred_nstep, y_pred_nEDA, y_pred_lasso], ['Stepwise', 'EDA', 'Lasso'], True)

