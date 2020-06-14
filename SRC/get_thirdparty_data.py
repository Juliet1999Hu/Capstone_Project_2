# ----------------------------------------------------------------------------
# File name: get_chain_store.py
#
# Created on: Jan. 09 2020
#
# by Julia Hu
#
# Description:
#
# 1) This module contains function to get the chain store data from s3 by each state
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
import csv
from datetime import datetime
import boto3
import io

def get_thirdparty_data(state,s3, bucket, key, data_attribute, config,datapulllogger):
    
    ################  Read chain_store_data file from S3 ############################################################################
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))

    ##load Selected columns 
    columnList = []
    for key,val in data_attribute.items():
        columnList.append(val)
    
    df = df[columnList]
    
    if(len(df) == 0):
        datapulllogger.error("No third party %s for %s is loaded in" % (key,state))
        
    return df[df['State'] == state]