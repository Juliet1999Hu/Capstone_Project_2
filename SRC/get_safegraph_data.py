# ----------------------------------------------------------------------------
# File name: get_safegraph_data.py
#
# Created on: Nov. 08 2019
#
# by Julia Hu
#
# Description:
#
# 1) This module contains function to get the safegraph_data from each state
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

def get_safegraph_data(state,config,datapulllogger):
    #first connect to Redshift with SQLALCHEMY
    POSTGRES_URL = config['Redshift']['POSTGRES_URL']
    POSTGRES_USER = config['Redshift']['POSTGRES_USER']
    POSTGRES_PW = config['Redshift']['POSTGRES_PW']
    POSTGRES_DB = config['Redshift']['POSTGRES_DB']
    
    DB_URL = 'postgresql+psycopg2://{user}:{pw}@{url}/{db}'.format(user=POSTGRES_USER,pw=POSTGRES_PW,url=POSTGRES_URL,db=POSTGRES_DB)
    engine = create_engine(DB_URL)
    
    ##load Selected CHD data as chd_df
    sql = config.get('SQL_query','get_safegraph_by_state',raw=True)
    safegraph_df = pd.read_sql_query(sql, engine, params={"sql_state":state,})
    
    engine.dispose()
    datapulllogger.info('safegraph data engine disconnected for %s' % state)
    if(len(safegraph_df) == 0):
        datapulllogger.error("No safegraph_data for %s is loaded in" % state)
        
    return safegraph_df