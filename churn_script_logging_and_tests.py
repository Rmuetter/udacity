"""
module to test the functions of churn_library.py

author: Robert
date: 01.12.2021
"""

import os
import logging
import churn_library as cl
import pytest
from churn_library import import_data

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return df

def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    perform_eda(df)
    path = "./images/eda"
    try:
        assert len(df)>0
        logging.info("SUCCESS: no empty dataframe provided")
    except AssertionError as err:
        logging.error("ERROR: empty dataframe provided")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                   'Income_Category', 'Card_Category'
                  ]
    try:
        df = encoder_helper(df, cat_columns)
        for col in cat_columns:
            assert col in df.columns
        logging.info("SUCCESS: encoded successfully the features")
    except AssertionError as err:
        logging.error("ERROR: could not encode the features")
        return err

    return df

def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:        
        X_train, X_test, y_train, y_test = perform_feature_engineering(df, "Churn")
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info("SUCCESS: performed feature engineering")
    except AssertionError as err:
        logging.error("ERROR: something went wrong in feature engineering")
        raise err

    return X_train, X_test, y_train, y_test


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        list_files = os.listdir("./images/results/")
        assert len(list_files) > 0
        logging.info("SUCCESS: result images found")
    except FileNotFoundError as err:
        logging.error("ERROR: no result images found")
        raise err
    
    try:
        list_files = os.listdir("./models/")
        assert len(list_files) > 0
        logging.info("SUCCESS: models found")
    except FileNotFoundError as err:
        logging.error("ERROR: no models found")
        raise err


if __name__ == "__main__":
    DATA = test_import(cl.import_data)
    test_eda(cl.perform_eda, DATA)
    DATA = test_encoder_helper(cl.encoder_helper, DATA)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        cl.perform_feature_engineering, DATA)
    test_train_models(cl.train_models, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)








