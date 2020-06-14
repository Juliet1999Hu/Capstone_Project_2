# ----------------------------------------------------------------------------
# File name: data_clean.py
#
# Created on: January. 02 2020
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

def clean_df(yelp_df, state, config,datacleanlogger): 
    """
    Description: create cleaned feature for ML algorithm
    Input: 1) yelp_df
           2) sysco_df
    Output: Return cleaned dataframes
           1) yelp_final_df
           2) sysco_df
    """
    ### First only obtain first records from that unique "safegraph_place_id"
    yelp_df = yelp_df.groupby(['id']).first().reset_index()
      
    ### Fetch right column names from config file to cleanse and match CHD date   
    ### This is used to avoid hard-coding 
    yelp_name = config['matching_attribute_Yelp']['attr_1']
    yelp_address_1 = config['matching_attribute_Yelp']['attr_2'] 
    yelp_city = config['matching_attribute_Yelp']['attr_3']
    yelp_state = config['matching_attribute_Yelp']['attr_4']
    yelp_zip = config['matching_attribute_Yelp']['attr_5']
    yelp_address_2 = config['matching_attribute_Yelp']['attr_6']
    yelp_address_3 = config['matching_attribute_Yelp']['attr_7'] 
    yelp_lat = config['matching_attribute_Yelp']['attr_10'] 
    yelp_lon = config['matching_attribute_Yelp']['attr_11'] 
    yelp_cat = config['matching_attribute_Yelp']['attr_12']
    
    ### Start to cleansing the data for Record_Linkage
    ############## clean up the new data ###################

    yelp_df["yelp_state_cleansed"] = yelp_df[yelp_state].str.lstrip().str.rstrip()
    yelp_df["yelp_city_cleansed"] = yelp_df[yelp_city].str.lstrip().str.rstrip().str.upper()
    
    #############One more cleansing process needs to be added for this data#####################################
    ####################remove any space between two city names************************************************
    yelp_df["yelp_city_cleansed"] = yelp_df["yelp_city_cleansed"].str.replace(" ","")
    
    ### Only leave the first 5 digits of the zip code
    yelp_df["yelp_zip_cleansed"] = yelp_df[yelp_zip].astype(str).apply(lambda x: x[:5] if x!= '' else x) 
    
     
    ### for name, remove the special character and space at the beginning and end
    yelp_df["yelp_name_cleansed"] = yelp_df[yelp_name].str.replace("[]\\?!\"\'#$%&(){}+*/:;,._`|~\\[<=>@\\^-]", "")  
    yelp_df["yelp_name_cleansed"] = yelp_df['yelp_name_cleansed'].str.lstrip().str.rstrip().str.upper()
    
    yelp_df["yelp_cat_cleansed"] = yelp_df[yelp_cat].str.replace("[]\\?!\"\'#$%&(){}+*/:;,._`|~\\[<=>@\\^-]", "")  
    yelp_df["yelp_cat_cleansed"] = yelp_df['yelp_cat_cleansed'].str.lstrip().str.rstrip().str.upper()
    
    yelp_df["yelp_lat_cleansed"] = yelp_df[yelp_lat].astype(float).apply(lambda x:round(x,6))
    yelp_df["yelp_lon_cleansed"] = yelp_df[yelp_lon].astype(float).apply(lambda x:round(x,6))
    
    def str_join(df, sep, *cols):
        from functools import reduce
        return reduce(lambda x, y: x.astype(str).str.cat(y.astype(str), sep=sep), [df[col] for col in cols])

    ### First combine two address columns together, address_1 and address_2
    yelp_df['final_address'] = str_join(yelp_df[[yelp_address_1, yelp_address_2]],' ','address_1','address_2')
    
    ### for address, remove the special character and space at the beginning and end
    yelp_df["yelp_address_cleansed"] = yelp_df['final_address'].str.replace("[]\\?!\"\'#$%&(){}+*/:;,._`|~\\[<=>@\\^-]", "")
    yelp_df["yelp_address_cleansed"] = yelp_df['yelp_address_cleansed'].str.lstrip().str.rstrip().str.upper()
    
    # Split the address into number and street
    yelp_df['addr_street'] = yelp_df.yelp_address_cleansed.str.replace("^(\\d*)\\s*", "").astype(str) 
    yelp_df['addr_number'] = yelp_df.yelp_address_cleansed.str.replace("\\s(.*)", "")    
    
    ### column List
    yelp_df.rename(columns={'yelp_name_cleansed': "CLEANSED_CUSTOMER_NAME", 'yelp_city_cleansed': 'CLEANSED_CITY','yelp_state_cleansed':'CLEANSED_STATE','yelp_zip_cleansed':'CLEANSED_ZIP','yelp_address_cleansed':'CLEANSED_ADDRESS_1','yelp_lat_cleansed':'CLEANSED_LATITUDE','yelp_lon_cleansed':'CLEANSED_LONGITUDE','yelp_cat_cleansed': 'CLEANSED_CATEGORY'},inplace=True)
    
    ### Drop a few riginal columns to save memory
    yelp_df.drop(yelp_df.iloc[:, 1:15], inplace = True, axis = 1) 
    
    return yelp_df   
 
