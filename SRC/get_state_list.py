# ----------------------------------------------------------------------------
# File name: get_state_list.py
#
# Created on: Sep. 26 2019
#
#  by Julia Hu
#
# Description:
#
# 1) This module contains function to get the state list 
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
from sqlalchemy.exc import SQLAlchemyError
import psycopg2
import pymssql

def get_state_list(config,datapulllogger):
    
    #first connect to Redshift with SQLALCHEMY
    POSTGRES_URL = config['Redshift_Yelp']['POSTGRES_URL']
    POSTGRES_USER = config['Redshift_Yelp']['POSTGRES_USER']
    POSTGRES_PW = config['Redshift_Yelp']['POSTGRES_PW']
    POSTGRES_DB = config['Redshift_Yelp']['POSTGRES_DB']
    
    DB_URL = 'postgresql+psycopg2://{user}:{pw}@{url}/{db}'.format(user=POSTGRES_USER,pw=POSTGRES_PW,url=POSTGRES_URL,db=POSTGRES_DB)
    engine = create_engine(DB_URL)
    
    
    #load stateList data from redshift into pandas dataframe, and create an engine, then close the engine
    df = pd.read_sql_query(config['SQL_query']['get_state_list_sql'],engine)
         
    engine.dispose()
    
    datapulllogger.info('StateList engine disconnected')
    
    stateList = df.state.sort_values()
    if(len(stateList) > 0):
        ## remove the NULL
        stateList = stateList[stateList.notnull()]
        ## remove the ""   
        stateList = stateList[stateList != ""]
    else: 
        datapulllogger.error("No state is loaded in")
        
    return stateList