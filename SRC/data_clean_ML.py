# ----------------------------------------------------------------------------
# File name: data_clean.py
#
# Created on: Nov. 11 2019
# by Julia Hu
#
# Description:
#
# 1) This module contains function to cleansing data
#
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
from functools import reduce
from sqlalchemy.exc import SQLAlchemyError
import psycopg2
import pymssql
from functools import reduce

def clean_df(safegraph_df,state, config,datacleanlogger): 
    """
    Description: create cleaned feature for ML algorithm
    Input: 1) safegraph_df
           
    Output: Return cleaned dataframes
           1) safegraph_final_df
           
    """
    ### First only obtain first records from that unique "safegraph_place_id"
    safegraph_df = safegraph_df.groupby(['safegraph_place_id']).first().reset_index()
      
    ### Fetch right column names from config file to cleanse and match CHD date   
    ### This is used to avoid hard-coding 
    safegraph_name = config['matching_attribute_Safegraph']['attr_1']
    safegraph_address = config['matching_attribute_Safegraph']['attr_2'] 
    safegraph_city = config['matching_attribute_Safegraph']['attr_3']
    safegraph_state = config['matching_attribute_Safegraph']['attr_4']
    safegraph_zip = config['matching_attribute_Safegraph']['attr_5']
    safegraph_lat = config['matching_attribute_Safegraph']['attr_8']
    safegraph_lon = config['matching_attribute_Safegraph']['attr_9']
    safegraph_cat = config['matching_attribute_Safegraph']['attr_10']
    
    safegraph_df['CLEANSED_PHONE'] = safegraph_df.phone_number.str.replace("-","")
    safegraph_df['CLEANSED_PHONE'] = safegraph_df['CLEANSED_PHONE'].apply(lambda x:x[2:12] if (x != None and x.startswith('+1')) else (x[:10] if x!=None else x)).apply(pd.to_numeric, errors='coerce')
    safegraph_df['CLEANSED_PHONE'].fillna(0,inplace=True)
    
    
    ############## clean up the chd data ###################
    safegraph_df["safegraph_state_cleansed"] = safegraph_df[safegraph_state].str.lstrip().str.rstrip()
    safegraph_df["safegraph_city_cleansed"] = safegraph_df[safegraph_city].str.lstrip().str.rstrip().str.upper()
    
    #############One more cleansing process needs to be added for this data#####################################
    ####################remove any space between two city names************************************************
    safegraph_df["safegraph_city_cleansed"] = safegraph_df["safegraph_city_cleansed"].str.replace(" ","")
    
    ### Only leave the first 5 digits of the zip code
    safegraph_df["safegraph_zip_cleansed"] = safegraph_df[safegraph_zip].astype(str).apply(lambda x: x[:5] if x!= '' else x) 
    
     
    ### for name, remove the special character and space at the beginning and end
    safegraph_df["safegraph_name_cleansed"] = safegraph_df[safegraph_name].str.replace("[]\\?!\"\'#$%&(){}+*/:;,._`|~\\[<=>@\\^-]", "")  
    safegraph_df["safegraph_name_cleansed"] = safegraph_df['safegraph_name_cleansed'].str.lstrip().str.rstrip().str.upper()
    
    safegraph_df["safe_cat_cleansed"] = safegraph_df[safegraph_cat].str.replace("[]\\?!\"\'#$%&(){}+*/:;,._`|~\\[<=>@\\^-]", "")  
    safegraph_df["safe_cat_cleansed"] = safegraph_df['safe_cat_cleansed'].str.lstrip().str.rstrip().str.upper()
    
    safegraph_df["safe_lat_cleansed"] = safegraph_df[safegraph_lat].astype(float).apply(lambda x:round(x,6))
    safegraph_df["safe_lon_cleansed"] = safegraph_df[safegraph_lon].astype(float).apply(lambda x:round(x,6))
    
    ### for address, remove the special character and space at the beginning and end
    safegraph_df["safegraph_address_cleansed"] = safegraph_df[safegraph_address].str.replace("[]\\?!\"\'#$%&(){}+*/:;,._`|~\\[<=>@\\^-]", "")
    safegraph_df["safegraph_address_cleansed"] = safegraph_df['safegraph_address_cleansed'].str.lstrip().str.rstrip().str.upper()
    
    # Split the address into number and street
    safegraph_df['addr_street'] = safegraph_df.safegraph_address_cleansed.str.replace("^(\\d*)\\s*", "").astype(str) 
    safegraph_df['addr_number'] = safegraph_df.safegraph_address_cleansed.str.replace("\\s(.*)", "")    
    
    ### column List
    safegraph_df.rename(columns={'safegraph_name_cleansed': "CLEANSED_CUSTOMER_NAME", 'safegraph_city_cleansed': 'CLEANSED_CITY','safegraph_state_cleansed':'CLEANSED_STATE','safegraph_zip_cleansed':'CLEANSED_ZIP','safegraph_address_cleansed': 'CLEANSED_ADDRESS_1', 'safe_lat_cleansed':'CLEANSED_LATITUDE','safe_lon_cleansed':'CLEANSED_LONGITUDE','safe_cat_cleansed': 'CLEANSED_CATEGORY'},inplace=True)
    
    ### Drop a few riginal columns to save memory
    safegraph_df.drop(safegraph_df.iloc[:, 1:15], inplace = True, axis = 1) 
    
    return safegraph_df
    

    
 
