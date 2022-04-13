# library doc string


# import libraries
import os
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report
os.environ['QT_QPA_PLATFORM']='offscreen'



def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
	try:
        df = pd.read_csv(pth)
        df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
        return df
    except FileNotFoundError:
        print("The file does not exist")
    except KeyError:
        print("Attrition_Flag column missing")

def perform_churn_hist(df):
    '''
    perform customer churn histplot on df and save figure to image folder
    input :
            df: pandas dataframe
    
    output: 
            None
    '''
    plt.figure(figsize=(20,10)) 
    df['Churn'].hist();
    plt.xlabel("0- employee not customer 1- customer churn")
    plt.ylabel("number of customers")
    plt.savefig('images/churn_hist.png')
    
def perform_age_hist(df):
    '''
    perform customer age histplot on df and save figure to image folder
    input :
            df: pandas dataframe
    
    output: 
            None
    '''
    plt.figure(figsize=(20,10)) 
    df['Customer_Age'].hist();
    plt.xlabel("Customer ages")
    plt.ylabel("Number of Customers")
    plt.savefig('images/age_hist.png')

def perform_marital_status_bar(df):
        '''
    perform marital status bar on df and save figure to image folder
    input :
            df: pandas dataframe
    
    output: 
            None
    '''
    plt.figure(figsize=(20,10)) 
    df.Marital_Status.value_counts('normalize').plot(kind='bar');
    plt.title("Marital status for customers")
    plt.ylabel("Portion of marital status customers")
    plt.legend()
    plt.savefig('images/marital_status_bar')

def perform_total_trans_dist(df):
    '''
    perform total trans customer on df and save figure to image folder
    input :
            df: pandas dataframe
    
    output: 
            None
    '''
    plt.figure(figsize=(20,10))
    sns.histplot(df['Total_Trans_Ct'], color="blue", label="100% Equities", kde=True, stat="density", linewidth=0)
    plt.savefig('images/total_trans.png')

def perform_heatmap_figure(df):
    '''
    perform heatmap on df and save figure to image folder
    input :
            df: pandas dataframe
    
    output: 
            None
    '''
    plt.figure(figsize=(20,10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig('images/heatmap.png')

def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe
    
    output:
            None
    '''
    try:
        perform_churn_hist(df)
        perform_age_hist(df)
        perform_marital_status_bar(df)
        perform_total_trans_dist(df)
        perform_heatmap_figure(df)
    except:
        print("Error in performing eda")
    

def encoder_helper(df, category_lst, response="_churn"):
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
    for col in cols:
        groups = df.groupby(category_lst).mean()['Churn']
        lst = []
            for val in df[category_lst]:
                lst.append(groups.loc[val])
        df[col+response] = lst
    return df

def perform_feature_engineering(df, response="_churn"):
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
    df = encoder_helper(df, ["Gender", "Education_Level", "Marital_Status", "Card_Category"], response)
    X_feature = df['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']
    y_feature = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X_feature, y_feature, test_size= 0.3, random_state=42)
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
    pass


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
    pass

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
    pass