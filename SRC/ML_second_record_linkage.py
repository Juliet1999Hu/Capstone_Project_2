# ----------------------------------------------------------------------------
# File name: main_record_linkage
#
# Created on: Dec. 31 2019
#
# by Julia Hu
#
# Description:
#
# 1) This module contains function for Record_Linkage
#
#    Major Changes: (1) Incorporate ML model   
#                    
#
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

def main_record_linkage(clean_safegraph_df,third_df,merged_cluster_df, state, config,algorithmlogger,outFolderPath,outfoldername): 
    """
    Description: create cleaned feature for ML algorithm
    Input: 1) clean_safegraph_df except these merged_cluster_df
           2) third_df
    Output: Return cleaned dataframes
           merged_cluster_df
    """   
#######################Further process here to remove already matched sysco and safegraph data for second match#########  
    
    clean_safegraph_df = clean_safegraph_df[~clean_safegraph_df.safegraph_place_id.isin(merged_cluster_df['safegraph_place_id'])].reset_index().drop('index',axis=1)
   
    one_yelp_df = third_df[~(third_df.yelp_id.isin(merged_cluster_df['yelp_id']))]
    one_chain_df = third_df[~(third_df.chainstore_ID.isin(merged_cluster_df['chainstore_ID']))]
    one_defi_df = third_df[~(third_df['Definitive ID'].isin(merged_cluster_df['Definitive ID']))]
    one_str_df = third_df[~(third_df['STR ID'].isin(merged_cluster_df['STR ID']))].reset_index().drop('index',axis=1)
    third_2ML_df = pd.concat([one_yelp_df,one_chain_df,one_defi_df,one_str_df],ignore_index=True,sort=False).drop_duplicates()
    
    ##################### ##################### ### START DATA COLUMNS SELECTION ############ ############ ############ ############ ############ ############
    columnList = []
    for key,val in config['matching_attribute_SYSCO_1'].items():
        columnList.append(val)
    
    Data1_df = third_2ML_df[columnList]
    
    Data2_df = clean_safegraph_df[columnList]
    
    ##################### ##################### ### OUTPUT COLUMN SELECTION ############ ############ ############ ############ ############ ############
    #***************************Select useful columns respectively for sysco, and chd****************************
    #******************************Read output column lists from config file*************************************
    #************************************************************************************************************
    third_party_output_list = []
    for key,val in config['third_party_output_list'].items():
        third_party_output_list.append(val)
    
    third_sel_df = third_2ML_df[third_party_output_list]
    
    safegraph_output_list = []
    for key,val in config['safegraph_output_list'].items():
        safegraph_output_list.append(val)
    
    safegraph_sel_df = clean_safegraph_df[safegraph_output_list]
   
    ##################### ##################### ### Create block list from config file ############ ############ ############ ############ ############ ############
    blockList_2 = []
    for key,val in config['blocking_attribute_2'].items():
        blockList_2.append(val)
        
    filename = config['ML_model']['stage2_model']
    EM_threshold = float(config['ML_threshold']['stage2_model'])
    chunk_size = int(config['ML_chunksize']['stage2_size'])
    
    ##################### ##################### ### Read matching attributes ############ ############ ############ #
    CLEANSED_CUSTOMER_NAME = config['matching_attribute_SYSCO_1']['attr_1']
    CLEANSED_CITY = config['matching_attribute_SYSCO_1']['attr_2'] 
    CLEANSED_STATE = config['matching_attribute_SYSCO_1']['attr_3']
    CLEANSED_ZIP = config['matching_attribute_SYSCO_1']['attr_4']
    addr_street = config['matching_attribute_SYSCO_1']['attr_5']
    addr_number = config['matching_attribute_SYSCO_1']['attr_6'] 
    CLEANSED_ADDRESS = config['matching_attribute_SYSCO_1']['attr_7']
    CLEANSED_LATITUDE = config['matching_attribute_SYSCO_1']['attr_9']
    CLEANSED_LONGITUDE = config['matching_attribute_SYSCO_1']['attr_10']
    CLEANSED_CATEGORY = config['matching_attribute_SYSCO_1']['attr_11']
    
    ### ***************************Start for RecordLinkage Matching*************************************************
    #create block linkage index to confine the search 
    indexer = recordlinkage.Index()
    indexer.block(blockList_2)
    
    candidate_links = indexer.index(Data1_df, Data2_df)
    
    Data1_df['addr_number'] = pd.to_numeric(Data1_df['addr_number'].copy(),errors='coerce')
    Data2_df['addr_number'] = pd.to_numeric(Data2_df['addr_number'].copy(),errors='coerce')
    
    #####************************* Record Comparison************************************************
    compare = recordlinkage.Compare()

    compare.string(CLEANSED_CUSTOMER_NAME, CLEANSED_CUSTOMER_NAME, method='jarowinkler', label = 'CLEANSED_CUST_NAME')
    compare.string(CLEANSED_ADDRESS, CLEANSED_ADDRESS, method='jarowinkler', label = 'CLEANSED_ADDRESS_1')
    compare.string(addr_street, addr_street, method='jarowinkler', label = 'addr_street')
    compare.numeric(addr_number,addr_number, method='linear', offset=0.0, scale=100, origin=0.0, label = 'addr_number')
    compare.exact(CLEANSED_LATITUDE, CLEANSED_LATITUDE, label = 'CLEANSED_LATITUDE')
    compare.exact(CLEANSED_LONGITUDE, CLEANSED_LONGITUDE, label = 'CLEANSED_LONGITUDE')
    compare.string(CLEANSED_CATEGORY, CLEANSED_CATEGORY, method='jarowinkler', label = 'CLEANSED_CATEGORY')
    
    match_df = pd.DataFrame()
    
     #####************************ The comparison vectors******************************************************
    for i in range(0,candidate_links.shape[0],chunk_size):
        df_final = pd.DataFrame()
        compare_vectors = compare.compute(candidate_links[i:i+chunk_size], Data1_df, Data2_df)     
        
        df = compare_vectors.reset_index()
   
        df_final = df_final.append(df, ignore_index=True)
        print("stage2 ML for chunk_size {}......".format(i))
        
        if (len(df_final) > 0):
            # Load model parameters from the saved pkl file 
            dtest = xgb.DMatrix(df_final.iloc[:,2:], missing = np.NAN)
            # ******************** Load Model Functions (& metrics) and Predict ***************                
            loaded_model = joblib.load(filename)
            pred_y = loaded_model.predict(dtest)
            df_final['third_party_match_probs'] = np.array(pred_y)
            
            match_df = match_df.append (df_final[(df_final['third_party_match_probs']>EM_threshold) | ((df_final['third_party_match_probs'] > 0.6) & (df_final['CLEANSED_CUST_NAME'] > 0.8) & (df_final['CLEANSED_ADDRESS_1'] > 0.75))][['level_0','level_1','third_party_match_probs']], ignore_index=True)
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
    
        merged_cluster_df_2 = one_third_df.merge(cluster2_df_final,how='inner',on=['level_0','level_1']).sort_values(by='third_party_match_probs',ascending=False).drop_duplicates()
        merged_cluster_df_2.drop(['index_x','index_y','level_0','level_1'], axis=1,inplace=True)
        
    else:
        cluster1_df = pd.DataFrame(data=None,columns=third_sel_df.columns)
        cluster1_df['third_party_match_probs'] = 0.4
        cluster2_df = pd.DataFrame(data=None,columns=safegraph_sel_df.columns)
        merged_cluster_df_2 = pd.concat([cluster1_df,cluster2_df], axis=1) 
    
    merged_cluster_df_2 = merged_cluster_df_2[['safegraph_place_id','yelp_id','chainstore_ID','Definitive ID','STR ID']]    
    merged_cluster_final_df = pd.concat([merged_cluster_df, merged_cluster_df_2],ignore_index=True, sort = False)
    
    ### *************************Produce final output dataframe**********************************************
    ### ********************************merged_Uncluster_df***************************************************
    ### ******************************************************************************************************        
    if (len(merged_cluster_final_df) < len(third_df)) and (len(merged_cluster_final_df) != 0):
        uncluster1_df_final = safegraph_sel_df[~safegraph_sel_df.safegraph_place_id.isin (merged_cluster_final_df['safegraph_place_id'])].reset_index()
        uncluster1_df_final['third_party_match_probs']=0.4        
        uncluster2_df = pd.DataFrame(data=None, columns=third_sel_df.columns)
        merged_Uncluster_df_2 = pd.concat([uncluster1_df_final,uncluster2_df], axis=1)
        merged_Uncluster_df_2.drop('index', axis=1,inplace=True)
        merged_Uncluster_df_2 = merged_Uncluster_df_2[['safegraph_place_id','yelp_id','chainstore_ID','Definitive ID','STR ID']]
        
        uncluster_yelp_df = third_sel_df[~(third_sel_df.yelp_id.isin(merged_cluster_final_df['yelp_id']))]
        uncluster_chain_df = third_sel_df[~(third_sel_df.chainstore_ID.isin(merged_cluster_final_df['chainstore_ID']))]
        uncluster_defi_df = third_sel_df[~(third_sel_df['Definitive ID'].isin(merged_cluster_final_df['Definitive ID']))]
        uncluster_str_df = third_sel_df[~(third_sel_df['STR ID'].isin(merged_cluster_final_df['STR ID']))].reset_index()
        uncluster_third_df = pd.concat([uncluster_yelp_df,uncluster_chain_df,uncluster_defi_df,uncluster_str_df],ignore_index=True,sort=False).drop_duplicates()
        uncluster_third_df['third_party_match_probs']=0.4        
        uncluster2_3df = pd.DataFrame(data=None, columns=safegraph_sel_df.columns)
        merged_Uncluster_3df_2 = pd.concat([uncluster_third_df,uncluster2_3df], axis=1)
        merged_Uncluster_3df_2.drop('index', axis=1,inplace=True)
        merged_Uncluster_3df_2 = merged_Uncluster_3df_2[['safegraph_place_id','yelp_id','chainstore_ID','Definitive ID','STR ID']]
        
        merged_Uncluster_df_2_final = pd.concat([merged_Uncluster_df_2,merged_Uncluster_3df_2],ignore_index=True,sort=False)
        
    elif len(merged_cluster_final_df) == 0:
        uncluster1_df_final = safegraph_sel_df.copy()
        uncluster1_df_final['third_party_match_probs'] = 0.4
        
        uncluster2_df = pd.DataFrame(data=None, columns=third_sel_df.columns)
        merged_Uncluster_df_2 = pd.concat([uncluster1_df_final,uncluster2_df], axis=1)
        merged_Uncluster_df_2 = merged_Uncluster_df_2[['safegraph_place_id','yelp_id','chainstore_ID','Definitive ID','STR ID']]
        
        uncluster_third_df = third_sel_df.copy()
        uncluster_third_df['third_party_match_probs'] = 0.4
        
        uncluster2_3df = pd.DataFrame(data=None, columns=safegraph_sel_df.columns)
        merged_Uncluster_3df_2 = pd.concat([uncluster_third_df,uncluster2_3df], axis=1)
        merged_Uncluster_3df_2 = merged_Uncluster_3df_2[['safegraph_place_id','yelp_id','chainstore_ID','Definitive ID','STR ID']]
        
        merged_Uncluster_df_2_final = pd.concat([merged_Uncluster_df_2,merged_Uncluster_3df_2],ignore_index=True,sort=False)
    else: 
        uncluster1_df = pd.DataFrame(data=None, columns=safegrph_sel_df.columns)
        uncluster1_df_final = uncluster1_df.copy()
        uncluster1_df_final['third_party_match_probs'] = 0.4
        
        uncluster2_df = pd.DataFrame(data=None, columns=third_sel_df.columns)
        merged_Uncluster_df_2_final = pd.concat([uncluster1_df_final,uncluster2_df], axis=1)
            
    return merged_cluster_final_df, merged_Uncluster_df_2_final
          
