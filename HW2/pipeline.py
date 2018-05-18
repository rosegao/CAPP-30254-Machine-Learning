### Classification Machine Learning Pipeline
### Author: Rose Gao
### Course: Machine Learning for Public Policy

import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import norm, lognorm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from sklearn import tree, linear_model
from sklearn.model_selection import train_test_split, KFold, cross_val_score


### LOAD DATA

def load_data(filename, verbose=False):
    '''
    Load data as a pandas df and print specs.
    Inputs: filename
    Returns: df
    '''
    df = pd.read_csv(filename, index_col=0)
    if verbose:
        print('Dataset shape:', df.shape)
        print('Dataset columns:', df.columns)
        print('Dataset types:', df.dtypes)
        print('')
    return df

def load_data_dictionary(filename):
    df = pd.read_excel('GiveMeSomeCredit/Data%20Dictionary.xls')
    pd.set_option('display.max_colwidth', -1)
    return df

### EXPLORE DATA

def histogram(df, target, bins, xmin, xmax, ymin, ymax):
    '''
    Create histogram of target variable
    '''
    plt.figure(figsize=(7, 5))
    n, bins, patches = plt.hist(df[target], bins=bins, alpha=0.75, color='grey')

    # plot histogram
    plt.xlabel(target)
    plt.ylabel('Probability')
    plt.title('Histogram of ' + target)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.grid(True)
    plt.show()

def create_log_target(target, xmin, xmax, ymin, ymax):
    # fit the normal distribution on log target
    mu, sigma = norm.fit(np.log(target))

    # create histogram of log target
    plt.figure(figsize=(7,5))
    n, bins, patches = plt.hist(np.log(target), bins=60, normed=True, alpha=0.75, color='Grey')

    # add fitted line
    y = mlab.normpdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'g--', linewidth=2)

    # plot histogram
    plt.xlabel('Log Target')
    plt.ylabel('Probability')
    plt.title('Histogram of Log Target: mu = {}, sigma = {}'.format(round(mu, 2), round(sigma, 2)))
    plt.axis([xmin, xmax, ymin, ymax])
    plt.grid(True)
    plt.show()

# if log target is better:
def log_target(df):
    target = df.iloc[:,0]
    df['log_target'] = np.log(target)
    df = df.drop(target)
    return df

def create_correlation_heatmap(df):
    # compute correlation matrix
    corr = df.corr()

    # generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # create figure and plot
    fig = plt.figure(figsize=(len(cont_feats), 5))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap=sns.diverging_palette(220, 10, as_cmap=True))
    plt.show()

def drop_correlated_variables(df, correlated_variables):
    df = df.drop(correlated_variables, axis=1)
    return df

def cap_outliers(df, outliers):
    for outlier, cap in outliers:
        df[outlier] = df[outlier].apply(lambda x: min(cap, x))
    return df

def iqr_outlier(var):
    q1, q3 = np.percentile(var, [25, 75])
    iqr = q3 - q1
    lower_fence = q1 - (iqr * 1.5)
    upper_fence = q3 + (iqr * 1.5)
    return np.where((var > upper_fence) | (var < lower_fence))

### PREPROCESS AND CLEAN DATA

def replace_outliers(df):
    '''
    This method is specific to removing outliers for homework 2.
    '''
    mask = df['age'] == 0
    df.loc[mask, 'age'] = train_df['age'].median()

    mask = df['NumberOfTime30-59DaysPastDueNotWorse'] > 95
    df.loc[mask, 'NumberOfTime30-59DaysPastDueNotWorse'] = 0

    mask = df['NumberOfTime60-89DaysPastDueNotWorse'] > 95
    df.loc[mask, 'NumberOfTime60-89DaysPastDueNotWorse'] = 0

    mask = df['NumberOfTimes90DaysLate'] > 95
    df.loc[mask, 'NumberOfTimes90DaysLate'] = 0

    return df

def fill_na_with_zero(df):
    df2 = df.copy(deep=True)
    null_vars = list(df2.columns[df2.isnull().any()])
    for i in null_vars:
        df2[i] = df2[i].fillna(0)
    return df2

def fill_na_with_median(df):
    df2 = df.copy(deep=True)
    null_vars = list(df.columns[df2.isnull().any()])
    for i in null_vars:
        df2[i] = df2[i].fillna(df2[i].median())
    return df2

def fill_na_with_neg_one(df):
    df2 = df.copy(deep=True)
    null_vars = list(df2.columns[df2.isnull().any()])
    for i in null_vars:
        df2[i] = df2[i].fillna(-1)
    return df2

### GENERATE FEATURES

def one_hot_encoding_categorical(df, target):
    # separate out categorical features
    cat_feats = [x for x in df.select_dtypes(include=['object']).columns if x not in [target]]

    # one hot encoding for categorical features
    binary_feats = pd.get_dummies(df[cat_feats])

    # concatenate binary features
    df = pd.concat([df, binary_feats], axis=1)
    df = df.drop(cat_feats, axis=1)
    return df

def split_data(df, target, test_size=0.3):
    y = df[target] # or np.log(df[target])
    X = df.drop([target], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    print('X train shape:', X_train.shape)
    print('y train shape:', y_train.shape)
    print('X test shape:', X_test.shape)
    print('y test shape:', y_test.shape)
    return X_train, X_test, y_train, y_test

### EVALUATE CLASSIFIERS

def calculate_cv(model, X_train, y_train, n_splits=10, seed=8):
    kfold = KFold(n_splits, random_state=seed)
    scores = cross_val_score(model, X_train, y_train, cv=kfold)
    print('Model average score:', round(scores.mean(), 3))