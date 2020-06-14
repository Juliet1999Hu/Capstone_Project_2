# ----------------------------------------------------------------------------
# File name: data_clean.py
#
# Created on: Jan. 13 2020
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

def clean_df(str_df,state, config,datacleanlogger): 
    """
    Description: create cleaned feature for ML algorithm
    Input: 1) str_df
           
    Output: Return cleaned dataframes
           1) str_final_df
           
    """
    ### First only obtain first records from that unique "safegraph_place_id"
    str_df = str_df.groupby(['STR Number']).first().reset_index()
      
 ### Fetch right column names from config file to cleanse and match CHD date   
    ### This is used to avoid hard-coding 
    str_name = config['matching_attribute_Str']['attr_1']
    str_address = config['matching_attribute_Str']['attr_2'] 
    str_city = config['matching_attribute_Str']['attr_3']
    str_state = config['matching_attribute_Str']['attr_4']
    str_zip = config['matching_attribute_Str']['attr_5']
    str_lat = config['matching_attribute_Str']['attr_8']
    str_lon = config['matching_attribute_Str']['attr_9']
    
    str_df['CLEANSED_PHONE'] = str_df['Phone'].astype(str).apply(lambda x:x[2:12] if (x != None and x.startswith('+1')) else (x[:10] if x!=None else x)).apply(pd.to_numeric, errors='coerce')
    str_df['CLEANSED_PHONE'].fillna(0,inplace=True)
    
    
    ############## clean up the chd data ###################
    str_df["str_state_cleansed"] = str_df[str_state].str.lstrip().str.rstrip()
    str_df["str_city_cleansed"] = str_df[str_city].str.lstrip().str.rstrip().str.upper()
    
    #############One more cleansing process needs to be added for this data#####################################
    ####################remove any space between two city names************************************************
    str_df["str_city_cleansed"] = str_df["str_city_cleansed"].str.replace(" ","")
    
    ### Only leave the first 5 digits of the zip code
    str_df["str_zip_cleansed"] = str_df[str_zip].astype(str).apply(lambda x: x[:5] if x!= '' else x) 
    
     
    ### for name, remove the special character and space at the beginning and end
    str_df["str_name_cleansed"] = str_df[str_name].str.replace("[]\\?!\"\'#$%&(){}+*/:;,._`|~\\[<=>@\\^-]", "")  
    str_df["str_name_cleansed"] = str_df['str_name_cleansed'].str.lstrip().str.rstrip().str.upper()
    
    str_df["str_cat_cleansed"] = 'Hotel'
    
    str_df["str_lat_cleansed"] = str_df[str_lat].astype(float).apply(lambda x:round(x,6))
    str_df["str_lon_cleansed"] = str_df[str_lon].astype(float).apply(lambda x:round(x,6))
    
    ### for address, remove the special character and space at the beginning and end
    str_df["str_address_cleansed"] = str_df[str_address].str.replace("[]\\?!\"\'#$%&(){}+*/:;,._`|~\\[<=>@\\^-]", "")
    str_df["str_address_cleansed"] = str_df['str_address_cleansed'].str.lstrip().str.rstrip().str.upper()
    
    # Split the address into number and street
    str_df['addr_street'] = str_df.str_address_cleansed.str.replace("^(\\d*)\\s*", "").astype(str) 
    str_df['addr_number'] = str_df.str_address_cleansed.str.replace("\\s(.*)", "")    
    
    ### column List
    str_df.rename(columns={'str_name_cleansed': "CLEANSED_CUSTOMER_NAME", 'str_city_cleansed': 'CLEANSED_CITY','str_state_cleansed':'CLEANSED_STATE','str_zip_cleansed':'CLEANSED_ZIP','str_address_cleansed': 'CLEANSED_ADDRESS_1', 'str_lat_cleansed':'CLEANSED_LATITUDE','str_lon_cleansed':'CLEANSED_LONGITUDE','str_cat_cleansed': 'CLEANSED_CATEGORY'},inplace=True)
    
    ### Drop a few riginal columns to save memory
    str_df.drop(str_df.iloc[:, 1:10], inplace = True, axis = 1) 
    
    return str_df
    

    
 
