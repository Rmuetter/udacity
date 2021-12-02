"""
module to predict customer churn by using two machine learning models

author: Robert
date: 30.11.2021
"""
#import several libs
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import pytest
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; 
sns.set()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.preprocessing import StandardScaler

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    # read file
    df = pd.read_csv(pth)
    return df

def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    
    # histogram for distribution of churn
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig("./images/eda/hist_churn.png")
    plt.close()
    
    # histogram of customer age
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig("./images/eda/hist_customer_age.png")
    plt.close()
    
    # amount of customer martial status
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar');
    plt.savefig("./images/eda/marital_status_count.png")
    plt.close()
    
    # distribution plot of total transaction count
    plt.figure(figsize=(20, 10))
    sns.distplot(df['Total_Trans_Ct']);
    plt.savefig("./images/eda/Total_Trans_Ct_dist.png")
    plt.close()
    
    # heatmap of correlation between cols
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig("./images/eda/heatmap.png")
    plt.close()

def encoder_helper(df, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for col in category_lst:
        # generate empty list for each feature and group the dataframe by this feature
        cat_column_lst = []
        cat_column_groups = df.groupby(col).mean()['Churn']
    
        # append the list with the mean churn
        for val in df[col]:
            cat_column_lst.append(cat_column_groups.loc[val])
        col_name= col+"_Churn"
        df[col_name] = cat_column_lst
    
    return df

def perform_feature_engineering(df):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    
    cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
    ]
    df = encoder_helper(df, cat_columns)

    keep_cols = ["Churn", 'Customer_Age', 'Dependent_count', 'Months_on_book',
                 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                 'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
                 'Income_Category_Churn', 'Card_Category_Churn']

    df = df[keep_cols]
    
    # define target
    y = df['Churn']
    
    # define features by droping the target
    X = df.drop(["Churn"], axis=1)

    scaler = StandardScaler()   
    X = scaler.fit_transform(X)
    X = pd.DataFrame(data=X)
 
    #proceed train-test-split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test 

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # generate classification reports
    plt.figure()
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)))  # approach improved by OP -> monospace!
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)))  # approach improved by OP -> monospace!
    plt.savefig('./images/results/rf_results.png')
    plt.close()
    
    plt.figure()
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_lr)))  # approach improved by OP -> monospace!
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_lr)))  # approach improved by OP -> monospace!
    plt.savefig('./images/results/rf_results.png')
    plt.close() 

def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90);
    
    # save plt in output path
    plt.savefig(output_pth)
    plt.close()

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    #define param grid
    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }
    
    #define grid search and fit random forest classifier
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # fit logistic regression model
    lrc.fit(X_train, y_train)

    # predictions for random forest
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # predictions for logistic regression
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    
    # plot ROC for logistic regression
    lrc_plot = plot_roc_curve(
        lrc, 
        X_test, 
        y_test)
    
    # plot ROC for logistic regression and random forest and save it
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(
        cv_rfc.best_estimator_, 
        X_test, 
        y_test, 
        ax=ax, 
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig("images/results/ROC_result.png")

    # save the models als pickle file
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    
    # save classification report
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    # store feature importances plot
    feature_importance_plot(
        cv_rfc.best_estimator_,
        X_train,
        './images/results/feature_importance_plot.png')
    
if __name__ == "__main__":
    # import sample dataframe
    df_bank = import_data("./data/bank_data.csv")

    # perform eda
    perform_eda(df_bank)

    # perform train-test-splot
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        df_bank)

    # model training and store results
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)