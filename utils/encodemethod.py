import base64
import importlib

import os
import streamlit as st
#EDA pkgs
import pandas as pd
#Viz pkgs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
import math
import scipy.stats as ss
import seaborn as sns
from sklearn import preprocessing
from collections import Counter
from datetime import date
from sklearn.model_selection import train_test_split
import category_encoders as ce


def file_selector(folder_path='././datasets'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox("Select Dataset from available datasets",filenames)
    filename = os.path.join(folder_path,selected_filename)
    df = pd.read_csv(filename,low_memory=False)
    return {"df":df, "filename":selected_filename}

def file_selector_1(folder_path='././datasets'):
    filenames_1 = os.listdir(folder_path)
    selected_filename_1 = st.selectbox("Select Predected Dataset from available datasets",filenames_1)
    filename_1 = os.path.join(folder_path,selected_filename_1)
    df_1 = pd.read_csv(filename_1,low_memory=False)
    return {"df_1":df_1, "filename_1":selected_filename_1}

def na_summary(df):
    df_null = df.isnull().sum()
    df_null = df_null.to_frame().reset_index()
    df_null = df_null.rename(columns={'index':'Feature', 0:'NA_count'})
    df_null['NA_percentage'] = df_null['NA_count']/df.shape[0]*100
    return df_null

def value_encoding(df,col_t,col_i):
    confusion_matrix = pd.crosstab(df[col_t],df[col_i])
    chi2 = ss.chi2_contingency(confusion_matrix)[3]
    confusion_matrix_array = np.asarray(confusion_matrix)
    confusion_matrix_array = ((confusion_matrix_array-chi2))/chi2
    confusion_matrix_array = pd.DataFrame(confusion_matrix_array, index=confusion_matrix.index,columns=confusion_matrix.columns)
    new_data_frame= df[[col_t,col_i]]
    values = []
    for (i,j) in zip(new_data_frame[col_t],new_data_frame[col_i]):
        values.append(confusion_matrix_array[j][i])
    new_data_frame[col_i] = values
    return new_data_frame

def conditional_entropy(df,col_t,col_i):
    # entropy of x given y
    y_counter = Counter(df[col_i].tolist())
    xy_counter = Counter(list(zip(df[col_t].tolist(),df[col_i].tolist())))
    total_occurrences = sum(y_counter.values())
    confusion_matrix = pd.crosstab(df[col_t],df[col_i])
    new_data_frame= df[[col_t,col_i]]
    values = []
    for xy in xy_counter.keys():
        p_xy =float(xy_counter[xy] / total_occurrences)
        p_y = y_counter[xy[1]] / total_occurrences
        entropy = p_xy * math.log(p_y/p_xy)
        confusion_matrix.loc[xy[0],xy[1]] = entropy
    for (i,j) in zip(df[col_t],df[col_i]):
        values.append(confusion_matrix[j][i])
    new_data_frame[col_i]= values
    return new_data_frame

def label_encoding (df,col):
    label_encoder = preprocessing.LabelEncoder()
    new_data_frame = df
    new_data_frame[col] = label_encoder.fit_transform(df[col])
    return new_data_frame

def conversion_rate(df,target_colums, valid_columns):
    return_copy = df.copy()
    n = len(return_copy)
    for col in valid_columns:
        if col in df.columns:
            tmp = df.groupby(col)[target_colums].agg(lambda x: x.sum() / (x.count() * n)).reset_index()
            tmp[col + '_cr'] = tmp[target_colums]
            tmp = tmp.drop([target_colums], axis=1)
            return_copy = pd.merge(return_copy, tmp, how ='left', on = col)
    return_copy = return_copy.drop(df.columns.to_list(), axis = 1)
    return return_copy

def Chi_Square(df,target_colums, valid_columns):
   return_copy = df.copy()
   for col in valid_columns:                
       if col in df.columns:
           observed_val = pd.crosstab(df[target_colums],df[col])
           expected_val =pd.DataFrame(ss.chi2_contingency(observed_val)[3],columns = observed_val.columns , index = observed_val.index)
           chi = (((observed_val - expected_val))/expected_val)
           chi.iloc[0] = chi.iloc[1]
           chi = chi.reset_index()
           chi_melt = chi.melt(id_vars =[target_colums], value_name = col+'_CS')
           return_copy = pd.merge(return_copy , chi_melt, on=[col,target_colums], how='left')
   return_copy = return_copy.drop(df.columns.to_list(), axis = 1)
   return return_copy

def one_hot_encoding(df, col):
    ohe = pd.get_dummies(df[[col]])
    return(ohe)

def na_imputation(df,col,impuation_value):
    if impuation_value == 'mean':
        df[col] = df[col].fillna(df[col].mean()) 
    if impuation_value == 'mode':
        mode = df[col].mode()
        df[col] = df[col].fillna(mode[0])
    if impuation_value == 'median':
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(impuation_value)
    return df[col]
    
def tranformation(df,col,tranformation_type):
    if tranformation_type == 'log':
        df[col] = np.log(df[col])
    if tranformation_type == 'log+1':
        df[col] = np.log(df[col]+1)
    if tranformation_type == 'sqaure':
        df[col] = np.square(df[col])
    if tranformation_type == 'sqrt':
        df[col] = np.sqrt(df[col])
    if tranformation_type == 'cuberoot':
        df[col] = np.cbrt(df[col])
    if tranformation_type == 'normalize':
        lower = df[col].min()
        upper = df[col].max()
        df[col] = ((df[col]-lower)/(upper-lower))
    if tranformation_type == 'exp':
        df[col] = np.exp(df[col])
    if tranformation_type == 'sigmoid':
        e = np.exp(1)
        df[col] = 1/(1+e**(-df[col]))
    if tranformation_type == 'tanh':
        df[col] = np.tanh(df[col])
    if tranformation_type == 'percentile linerization':
        size = len(df[col])-1
        df[col] = df[col].rank(method='min').apply(lambda x: (x-1)/size)
    return df[col]

def savedataframe(new_df):
    st.text("Would you like to save new dataframe")
    filename =  st.text_input("file name:")
    if st.button("save"):
        new_df.to_csv('././datasets/'+filename+'.csv',index=False)

def savedataframe_1(new_df,a,b):
    st.text("Would you like to save new dataframe")
    filename =  st.text_input(a)
    if st.button(b):
        new_df.to_csv('././datasets/'+filename+'.csv',index=False)

def outlier(df,column,max_out,min_out,action):
    if action == 'replace':
        lower_elements_index = df[column] < min_out
        
        upper_elements_index = df[column] > max_out
        
        df[column][lower_elements_index] = min_out
        df[column][upper_elements_index] = max_out
    else:
        true_index = (min_out < df[column].values) & (max_out > df[column].values)
        
        false_index = ~true_index
        
        drop_index = list(df[false_index].index)
        
        df = df.drop(index = drop_index)
    return df

def date_features(data,column,choise):
    #Transform string to date
    data['date'] = pd.to_datetime(data[column], format="%d-%m-%Y")
    
    if (choise == 'TimePeriod'):
        #Extracting passed years since the date
        data['passed_years'] = date.today().year - data['date'].dt.year

        #Extracting passed months since the date
        data['passed_months'] = (date.today().year - data['date'].dt.year) * 12 + date.today().month - data['date'].dt.month

        #Extracting passed days since the date
        data['passed_days'] = pd.Timestamp(date.today()) - data['date']
        data['passed_days'] = data['passed_days'].astype(str).str.extract('(.*?) days')

        return data[['passed_years','passed_months','passed_days']]
    else:
        #Extracting the weekday name of the date
        data['day_name'] = data['date'].dt.day_name()
        return data['day_name']

def cat_encode(df,col_t,col_i,encode_type):
    # Note: All these techniques need the target variable to be label encoded
    # Classic Encoders
    if encode_type == 'BinaryEncoder':
        ce_fit = ce.BinaryEncoder(cols=[col_i])
        df_encode = ce_fit.fit_transform(df[col_i],df[col_t])
    if encode_type == 'BaseNEncoder':
        ce_fit = ce.BaseNEncoder(cols=[col_i])
        df_encode = ce_fit.fit_transform(df[col_i],df[col_t])
        
    # Contrast Encoders
    if encode_type == 'HelmertEncoder':
        ce_fit = ce.HelmertEncoder(cols=[col_i])
        df_encode = ce_fit.fit_transform(df[col_i],df[col_t])
        if 'intercept' in df_encode.columns:
            df_encode = df_encode.drop(columns=['intercept'])
    if encode_type == 'SumEncoder':
        ce_fit = ce.SumEncoder(cols=[col_i])
        df_encode = ce_fit.fit_transform(df[col_i],df[col_t])
        if 'intercept' in df_encode.columns:
            df_encode = df_encode.drop(columns=['intercept'])
    if encode_type == 'BackwardDifferenceEncoder':
        ce_fit = ce.BackwardDifferenceEncoder(cols=[col_i])
        df_encode = ce_fit.fit_transform(df[col_i],df[col_t])
        if 'intercept' in df_encode.columns:
            df_encode = df_encode.drop(columns=['intercept'])
    if encode_type == 'PolynomialEncoder':
        ce_fit = ce.PolynomialEncoder(cols=[col_i])
        df_encode = ce_fit.fit_transform(df[col_i],df[col_t])
        if 'intercept' in df_encode.columns:
            df_encode = df_encode.drop(columns=['intercept'])

    # Bayesian Encoders
    if encode_type == 'TargetEncoder':
        ce_fit = ce.TargetEncoder(cols=[col_i])
        df_encode = ce_fit.fit_transform(df[col_i],df[col_t])
    if encode_type == 'LeaveOneOutEncoder':
        ce_fit = ce.LeaveOneOutEncoder(cols=[col_i])
        df_encode = ce_fit.fit_transform(df[col_i],df[col_t])
    if encode_type == 'WOEEncoder':
        ce_fit = ce.WOEEncoder(cols=[col_i])
        df_encode = ce_fit.fit_transform(df[col_i],df[col_t])
    if encode_type == 'CatBoostEncoder':
        ce_fit = ce.CatBoostEncoder(cols=[col_i])
        df_encode = ce_fit.fit_transform(df[col_i],df[col_t])
    if encode_type == 'MEstimateEncoder':
        ce_fit = ce.MEstimateEncoder(cols=[col_i])
        df_encode = ce_fit.fit_transform(df[col_i],df[col_t])
    if encode_type == 'JamesSteinEncoder':
        ce_fit = ce.JamesSteinEncoder(cols=[col_i])
        df_encode = ce_fit.fit_transform(df[col_i],df[col_t])
    
    new_columns = [i+"_"+encode_type for i in df_encode.columns]
    df_encode = df_encode.set_axis(new_columns, axis=1, inplace=False)
    return df_encode