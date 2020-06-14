# ----------------------------------------------------------------------------
# File name: data_clean_definitive.py
#
# Created on: Jan. 13 2019
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

def clean_df(defi_df,state, config,datacleanlogger): 
    """
    Description: create cleaned feature for ML algorithm
    Input: 1) defi_df
           
    Output: Return cleaned dataframes
           1) defi_final_df
           
    """
    ### First only obtain first records from that unique "safegraph_place_id"
    defi_df = defi_df.groupby(['Definitive ID']).first().reset_index()
      
    ### Fetch right column names from config file to cleanse and match CHD date   
    ### This is used to avoid hard-coding 
    defi_name = config['matching_attribute_Definitive']['attr_1']
    defi_address = config['matching_attribute_Definitive']['attr_2'] 
    defi_address_1 = config['matching_attribute_Definitive']['attr_3'] 
    defi_city = config['matching_attribute_Definitive']['attr_4']
    defi_state = config['matching_attribute_Definitive']['attr_5']
    defi_zip = config['matching_attribute_Definitive']['attr_6']
    defi_lat = config['matching_attribute_Definitive']['attr_9']
    defi_lon = config['matching_attribute_Definitive']['attr_10']
    defi_cat = config['matching_attribute_Definitive']['attr_11']
    
    defi_df['CLEANSED_PHONE'] = defi_df.Phone.str.replace(".", "")
    defi_df['CLEANSED_PHONE'] = defi_df['CLEANSED_PHONE'].astype(str).apply(lambda x:x[2:12] if (x != None and x.startswith('+1')) else (x[:10] if x!=None else x)).apply(pd.to_numeric, errors='coerce')
    defi_df['CLEANSED_PHONE'].fillna(0,inplace=True)
    
    ############## clean up the chd data ###################
    defi_df["defi_state_cleansed"] = defi_df[defi_state].str.lstrip().str.rstrip()
    defi_df["defi_city_cleansed"] = defi_df[defi_city].str.lstrip().str.rstrip().str.upper()
    
    #############One more cleansing process needs to be added for this data#####################################
    ####################remove any space between two city names************************************************
    defi_df["defi_city_cleansed"] = defi_df["defi_city_cleansed"].str.replace(" ","")
    
    ### Only leave the first 5 digits of the zip code
    defi_df["defi_zip_cleansed"] = defi_df[defi_zip].astype(str).apply(lambda x: x[:5] if x!= '' else x) 
    
     
    ### for name, remove the special character and space at the beginning and end
    defi_df["defi_name_cleansed"] = defi_df[defi_name].str.replace("\([^()]*\)", "")
    defi_df["defi_name_cleansed"] = defi_df["defi_name_cleansed"].str.replace("[]\\?!\"\'#$%&(){}+*/:;,._`|~\\[<=>@\\^-]", "")  
    defi_df["defi_name_cleansed"] = defi_df['defi_name_cleansed'].str.lstrip().str.rstrip().str.upper()
    
    defi_df["defi_cat_cleansed"] = defi_df[defi_cat].str.replace("[]\\?!\"\'#$%&(){}+*/:;,._`|~\\[<=>@\\^-]", "") 
    defi_df["defi_cat_cleansed"] = defi_df['defi_cat_cleansed'].str.lstrip().str.rstrip().str.upper()
    
    defi_df["defi_lat_cleansed"] = defi_df[defi_lat].astype(float).apply(lambda x:round(x,6))
    defi_df["defi_lon_cleansed"] = defi_df[defi_lon].str.strip('()')
    defi_df["defi_lon_cleansed"] = defi_df["defi_lon_cleansed"].astype(float).apply(lambda x:round(x,6))
    
    ### for address, remove the special character and space at the beginning and end
    defi_df["defi_address_cleansed"] = defi_df[defi_address].str.replace("[]\\?!\"\'#$%&(){}+*/:;,._`|~\\[<=>@\\^-]", "")
    defi_df["defi_address_cleansed"] = defi_df['defi_address_cleansed'].str.lstrip().str.rstrip().str.upper()
    
    # Split the address into number and street
    defi_df['addr_street'] = defi_df.defi_address_cleansed.str.replace("^(\\d*)\\s*", "").astype(str) 
    defi_df['addr_number'] = defi_df.defi_address_cleansed.str.replace("\\s(.*)", "")    
    
    ### column List
    defi_df.rename(columns={'defi_name_cleansed': "CLEANSED_CUSTOMER_NAME", 'defi_city_cleansed': 'CLEANSED_CITY','defi_state_cleansed':'CLEANSED_STATE','defi_zip_cleansed':'CLEANSED_ZIP','defi_address_cleansed': 'CLEANSED_ADDRESS_1','defi_lat_cleansed':'CLEANSED_LATITUDE','defi_lon_cleansed':'CLEANSED_LONGITUDE','defi_cat_cleansed': 'CLEANSED_CATEGORY'},inplace=True)
    
    ### Drop a few riginal columns to save memory
    defi_df.drop(defi_df.iloc[:, 1:11], inplace = True, axis = 1) 
    
    return defi_df
    

    
 
