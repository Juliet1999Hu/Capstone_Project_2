# ----------------------------------------------------------------------------
# File name: UPLOAD_2_S3
#
# Created on: Jan. 28 2020
#
# by Julia Hu
#
# Description:
#
# 1) This module contains function for How to Upload output file to S3
#
#       
#
# -----------------------------------------------------------------------------
#first load in all necessary librares 
import os,sys
import boto3
from botocore.exceptions import ClientError
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import csv
from functools import reduce
import awswrangler

def upload_file(merged_cluster_df,merged_Uncluster_df,state,outFolderPath,outfoldername,logsFolderPath, logfoldername,config,outputlogger):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    final_df = pd.concat([merged_cluster_df, merged_Uncluster_df], ignore_index=True,sort=False)
    final_df["match ID"] = final_df.reset_index()["index"].values
    final_df['match ID'] = final_df['match ID'].apply(lambda x: state+str(x+1).zfill(8))
    
    s3_bucket = config['S3_param']['S3Bucket_name']  
    output_bucket_folder = config['S3_param']['out_bucket_folder']
    s3_url = config['S3_param']['S3_URL']+'/'+'SVOC_CUST_MDM_DIM'
    log_bucket_folder = config['S3_param']['log_bucket_folder']
    logtimestamp = datetime.now().strftime('%m-%d-%y')
    log_bucket_sub_folder = 'Log_'+logtimestamp
    
    ### *************************Write out clustered ssco and chd data by state**********************************************
    ### *************************Write out UNclustered ssco and chd data by state*************************
    merged_file_name = 'third_party_merged_%s.csv' % (state)
    
    final_df.to_csv(outFolderPath+outfoldername+'/'+ merged_file_name, header=True, index=False)
    
        # Upload the file
    s3_client = boto3.client('s3')
    
    # If S3 object_name was not specified, use file_name
    
    dest_file_name = merged_file_name
    try:
        response1 = s3_client.upload_file(outFolderPath+outfoldername+'/'+merged_file_name, s3_bucket, '%s/%s' % (output_bucket_folder,dest_file_name))
         
    except ClientError as e:
        outputlogger.error(e)
        return False
     
    #***************************** enumerate local log files recursively**************************************
    local_log_directory = logsFolderPath+logfoldername
    
    for filename in os.listdir(local_log_directory):
        
        dest_file_name = filename
        try:
            response = s3_client.upload_file(local_log_directory + '/'+ filename, s3_bucket, '%s/%s/%s' %(log_bucket_folder,log_bucket_sub_folder,dest_file_name))
            
        except ClientError as e:
               outputlogger.error(e)
               return False
    
    return True


    
 
