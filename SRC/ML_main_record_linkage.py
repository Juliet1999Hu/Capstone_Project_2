# ----------------------------------------------------------------------------
# File name: main_record_linkage
#
# Created on: Dec. 30 2019
#
# by Julia Hu
#
# Description:
#
# 1) This module contains function for Record_Linkage
#
#    Major changes including load ML model     
#                  filter CONCAT_ID, and Safegraph_place_id to remove multiple matches for one id 
# -----------------------------------------------------------------------------
#first load in all necessary librares 
import os,sys
import logging
import imp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import csv
import recordlinkage
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import psycopg2
import pymssql
from functools import reduce
import xgboost as xgb
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import joblib

def main_record_linkage(clean_safegraph_df,third_df,state, config,algorithmlogger,outFolderPath,outfoldername): 
    """
    Description: create cleaned feature for ML algorithm
    Input: 1) clean_chd_df
           2) third_df
    Output: Return cleaned dataframes
           merged_cluster_df
    """
    ##################### ##################### ### START DATA COLUMNS SELECTION ############ ############ ############ ############ ############ ############
    columnList = []
    for key,val in config['matching_attribute_third_party'].items():
        columnList.append(val)
    
    Data1_df = third_df[columnList]
    Data2_df = clean_safegraph_df[columnList]
    
    ##################### ##################### ### OUTPUT COLUMN SELECTION ############ ############ ############ ############ ############ ############
    #***************************Select useful columns respectively for sysco, and chd****************************
    #******************************Read output column lists from config file*************************************
    #************************************************************************************************************
    third_party_output_list = []
    for key,val in config['third_party_output_list'].items():
        third_party_output_list.append(val)
    
    third_sel_df = third_df[third_party_output_list]
    
    safegraph_output_list = []
    for key,val in config['safegraph_output_list'].items():
        safegraph_output_list.append(val)
    
    safegraph_sel_df = clean_safegraph_df[safegraph_output_list]
    
    ##################### ##################### ### Create block list from config file ############ ############ ############ ############ ############ ############
    blockList = []
    for key,val in config['blocking_attribute'].items():
        blockList.append(val)
        
    filename = config['ML_model']['stage1_model']
    EM_threshold = float(config['ML_threshold']['stage1_model'])    
    
    ##################### ##################### ### Read matching attributes ############ ############ ############ #
    CLEANSED_CUSTOMER_NAME = config['matching_attribute_third_party']['attr_1']
    CLEANSED_CITY = config['matching_attribute_third_party']['attr_2'] 
    CLEANSED_STATE = config['matching_attribute_third_party']['attr_3']
    CLEANSED_ZIP = config['matching_attribute_third_party']['attr_4']
    addr_street = config['matching_attribute_third_party']['attr_5']
    addr_number = config['matching_attribute_third_party']['attr_6'] 
    CLEANSED_PHONE = config['matching_attribute_third_party']['attr_7']
    CLEANSED_LATITUDE = config['matching_attribute_third_party']['attr_8']
    CLEANSED_LONGITUDE = config['matching_attribute_third_party']['attr_9']
    CLEANSED_CATEGORY = config['matching_attribute_third_party']['attr_10']
    
    ### ***************************Start for RecordLinkage Matching*************************************************
    #create block linkage index to confine the search 
    indexer = recordlinkage.Index()
    indexer.block(blockList)
    
    candidate_links = indexer.index(Data1_df, Data2_df)
    
    # Record Comparison
    compare = recordlinkage.Compare()

    compare.string(CLEANSED_CUSTOMER_NAME, CLEANSED_CUSTOMER_NAME, method='jarowinkler', label = 'CLEANSED_CUST_NAME')
    compare.string(addr_street, addr_street, method='jarowinkler', label = 'addr_street')
    compare.numeric(CLEANSED_PHONE, CLEANSED_PHONE, method='linear', offset=0.0, scale=100, origin=0.0, label = 'CLEANSED_PHONE')
    compare.exact(CLEANSED_LATITUDE, CLEANSED_LATITUDE, label = 'CLEANSED_LATITUDE')
    compare.exact(CLEANSED_LONGITUDE, CLEANSED_LONGITUDE, label = 'CLEANSED_LONGITUDE')
    compare.string(CLEANSED_CATEGORY, CLEANSED_CATEGORY, method='jarowinkler', label = 'CLEANSED_CATEGORY')
    
    # The comparison vectors
    compare_vectors = compare.compute(candidate_links, Data1_df, Data2_df)

    if (len(compare_vectors) > 0):
        df = compare_vectors.reset_index()
                
        # Load model parameters from the saved pkl file 
        dtest = xgb.DMatrix(df.iloc[:,2:], missing = np.NAN)
        # ******************** Load Model Functions (& metrics) and Predict ***************                
        loaded_model = joblib.load(filename)
        pred_y = loaded_model.predict(dtest)
        
        df['third_party_match_probs'] = np.array(pred_y)
#        match_df = df[df['third_party_match_probs']>EM_threshold][['level_0','level_1','third_party_match_probs']].sort_values(by = ['third_party_match_probs'],ascending=False)
        match_df = df[df['third_party_match_probs']>EM_threshold].sort_values(by = ['third_party_match_probs'],ascending=False)
        
    else:
        columns = ['level_0','level_1', 'third_party_match_probs']
        match_df = pd.DataFrame(columns=columns)
     
    ### *************************Produce final output dataframe**********************************************
    ### *************************one for merged_cluster_df, and one for merged_Uncluster_df*************************
    ### ******************************************************************************************************
    
    if len(match_df)>0:
        cluster1_df = third_sel_df.iloc[match_df['level_0']].reset_index().sort_values(by = ['index'])
        cluster1_df_final = cluster1_df.merge(match_df,how='inner', left_on = 'index', right_on = 'level_0' ).drop_duplicates()
        
        ### First combine four id columns together
        one_yelp_df = cluster1_df_final.sort_values(by=['yelp_id','third_party_match_probs'],ascending=False).groupby(['yelp_id']).first().reset_index()
        one_chainstore_df = cluster1_df_final.sort_values(by=['chainstore_ID','third_party_match_probs'],ascending=False).groupby(['chainstore_ID']).first().reset_index()
        one_defi_df = cluster1_df_final.sort_values(by=['Definitive ID','third_party_match_probs'],ascending=False).groupby(['Definitive ID']).first().reset_index()
        one_str_df = cluster1_df_final.sort_values(by=['STR ID','third_party_match_probs'],ascending=False).groupby(['STR ID']).first().reset_index()
        one_third_df = pd.concat([one_yelp_df,one_chainstore_df,one_defi_df,one_str_df],ignore_index=True,sort=False)
        
        cluster2_df = safegraph_sel_df.iloc[match_df['level_1']].reset_index()
        cluster2_df_final = cluster2_df.merge(match_df,how='inner', left_on = 'index', right_on = 'level_1' ).drop_duplicates()
        cluster2_df_final.drop('third_party_match_probs',axis=1,inplace=True)
    
        merged_cluster_df = one_third_df.merge(cluster2_df_final,how='inner',on=['level_0','level_1']).sort_values(by='third_party_match_probs',ascending=False).drop_duplicates()
        merged_cluster_df.drop(['index_x','index_y','level_0','level_1'], axis=1,inplace=True)
        
    else:
        cluster1_df = pd.DataFrame(data=None,columns=third_sel_df.columns)
        cluster1_df['third_party_match_probs'] = 0.4
        cluster2_df = pd.DataFrame(data=None,columns=safegraph_sel_df.columns)
        merged_cluster_df = pd.concat([cluster1_df,cluster2_df], axis=1) 
         
    merged_cluster_df = merged_cluster_df[['safegraph_place_id','yelp_id','chainstore_ID','Definitive ID','STR ID']] 
    return merged_cluster_df
   