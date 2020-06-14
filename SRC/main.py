# ----------------------------------------------------------------------------
# File name: Third Party Operator Matching Algorithm
#
# Author: Julia Hu
#
# Created on: 9th January 2020
#
# Important information: After ML, the final algorithm was implemented as a pickle file 
#
#
# Description:
#
# 1) This is initial exploration for Third Part Operator Matching Algorithm
#
# 
# 2) Steps:
#    a. Create File folder path and log files for specific sub modules
#    b. Read State list from Yelp database in Redshift
#    c. Read Yelp data from Redshift
#    d. Read safegraph data from Redshift
#    e. read chian_store_data
#    f. Read in definitive data
#    g. Read in str data, hotel data
#    e. Import data clansing module and pass the input data to clean the data
#    f. Import model building module and pass the feature engineered dataset (Apply threshold of 90% similarity to match different records together)
#    g. Write clustered and unclstered data Drop it to S3
#
# -----------------------------------------------------------------------------

def main():
    #first load in all necessary librares 
    import os,sys
    import logging
    from configparser import ConfigParser
    import importlib.util
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
    import boto3
    from botocore.exceptions import ClientError
    import xgboost as xgb
    from tqdm import tqdm
    from sklearn.metrics import accuracy_score
    import joblib
    import awswrangler
    #######################################################################################
    ### (a.1) Setup MasterpathFolder and other file folder's paths. Read in Configure file
    #######################################################################################
    # Be careful, you need to change this to __file__, not '__file__'
    masterFolderPath = os.path.dirname(__file__)
    # Adding helpers to path
    sys.path.append(masterFolderPath) 
    # setup other folders paths
    outFolderPath = masterFolderPath+'output/'
    logsFolderPath = masterFolderPath+'logs/'
    #set up configuration file path, and start to read configuration file in
    configFilePath = (os.path.join(masterFolderPath, 'ML_matching_config.ini'))
    # Create a folder for logging
    logtimestamp = datetime.now().strftime('%d_%b_%Y')
    logfoldername = 'Log_'+logtimestamp
    outfoldername = 'Output_'+logtimestamp
    
    # Check if folder exists otherwise create one
    if not os.path.exists(logsFolderPath+logfoldername+'/'):
        os.makedirs(logsFolderPath+logfoldername+'/')
    
    # Check if folder for outputexists otherwise create one
    if not os.path.exists(outFolderPath+outfoldername+'/'):
        os.makedirs(outFolderPath+outfoldername+'/')
    
    # log sub directory
    logSubDirectory = logsFolderPath+logfoldername+'/'
    
    # Setting a log files for all sub processes
    master_logfilename = '0.MasterLog'+'_'+logtimestamp+'.log'
    datapull_logfilename = '1.DataPull'+'_'+logtimestamp+'.log'
    dataclean_logfilename = '2.dataclean'+'_'+logtimestamp+'.log'
    algorithm_logfilename = '3.algorithm'+'_'+logtimestamp+'.log'
    output_logfilename = '4.WriteCsv'+'_'+logtimestamp+'.log'
    
    handlerlist = []  # Contains list of open file handlers that is used to close them later
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%a, %d %b %Y')
    
    # Setting up logger function
    def setup_logger(name, folpath, log_file, level=logging.DEBUG):
        """
        Function setup as many loggers as you want
        :param name : Name of the log handler
        :param folpath : Folder path where log file is to be created
        :param log_file : name of the log file
        :param level : debug level
        :return logger : logger 
        """
        handler = logging.FileHandler(folpath+log_file)
        handler.setFormatter(formatter)
        
        handlerlist.append(handler)
        
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)    
        return logger
    
    # Setup loggers for all sub processes
    masterlogger = setup_logger('master_logger', logSubDirectory, master_logfilename)
    datapulllogger = setup_logger('data_pull', logSubDirectory, datapull_logfilename)
    datacleanlogger = setup_logger('data_clean', logSubDirectory, dataclean_logfilename)
    algorithmlogger = setup_logger('algorithm', logSubDirectory, algorithm_logfilename)
    outputlogger = setup_logger('WriteCsv', logSubDirectory, output_logfilename)
    ###  ********************************************************************************************
    ### (a.2) Read in Configuration file with ConfigParser library***********************************
    ###**********************************************************************************************
    #read configuration file with ConfigParser, and write error message to log file
    config = ConfigParser()
    config.read(configFilePath)
    # Checking if Process params JSON is imported successfully
    if config.read(configFilePath) is not None:
        masterlogger.info("Configuration file imported successfully ")
    else:
        masterlogger.info("Issue with configuration file import")
    ###  ********************************************************************************************
    ### (b) This method is used to retrive CHD data from Redshift***********************************
    ###**********************************************************************************************
    # ****************Read stateList ****************
    # Start a log message
    masterlogger.info("StateList Data Pull Started")
    
    # Load function to pull data from redshift
    spec = importlib.util.spec_from_file_location("state_list", "./get_state_list.py")
    state_list = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(state_list)
    
    stateLs = state_list.get_state_list(config,datapulllogger)
    
    if len(stateLs)==0:
        df = pd.DataFrame(data=None) 
        merged_Safegraph_file_name = 'third_party_merged.csv'   
        df.to_csv(outFolderPath+outfoldername+'/'+ merged_Safegraph_file_name, header=True, index=False)
        masterlogger.info("No state in third-party records")
        
    
    ##################################Start to loop through all 51 states#############################
    ##################################################################################################
    for state in stateLs:
        print(state)
        ###Start to explore the data from Safegraph in State CA and TX. Get a feeling of total number of cities, within CA and TX. 
        # ****************Read chd data ****************
        # Load function to pull data from chd database
        spec = importlib.util.spec_from_file_location("safegraph_data","./get_safegraph_data.py")
        safegraph_data = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(safegraph_data)
        
        safegraph_df = safegraph_data.get_safegraph_data(state,config,datapulllogger)

        ###  ********************************************************************************************
        ### ### c. Read in CHD data from Redshift********************************************************
        ### *********************************************************************************************
        # Load function to pull data from chd database
        spec = importlib.util.spec_from_file_location("yelp_data", "./get_yelp_data.py")
        yelp_data = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(yelp_data)
    
        yelp_df = yelp_data.get_yelp_data(state,config,datapulllogger)

        # ****************Read third party data ****************
        # Load function to pull data from sysco database
        mod = imp.load_source("thirdparty_data","./get_thirdparty_data.py")
        
        import thirdparty_data
        s3 = boto3.client('s3')
        bucket = config['Third_party_S3_param']['bucket']
        key_chainstore = config['Third_party_S3_param']['key_chainstore']
        chainstore_attribute = config['chainstore_attribute']
        
        chain_df = thirdparty_data.get_thirdparty_data(state,s3, bucket, key_chainstore, chainstore_attribute, config,datapulllogger)
        chain_df['Industry Name'].apply(lambda x: x.split(' ',1)[1])
        
        #********read in assitant living facility
        key_definitive_1 = config['Third_party_S3_param']['key_definitive_1']
        definitive_1_attribute = config['definitive_1_attribute']
        defi_1_df = thirdparty_data.get_thirdparty_data(state,s3, bucket, key_definitive_1, definitive_1_attribute, config,datapulllogger)
        
        #********read in nursing home
        key_definitive_2 = config['Third_party_S3_param']['key_definitive_2']
        definitive_2_attribute = config['definitive_2_attribute']
        defi_2_df = thirdparty_data.get_thirdparty_data(state,s3, bucket, key_definitive_2, definitive_2_attribute, config,datapulllogger)
        
        #********read in hospitals**********
        key_definitive_3 = config['Third_party_S3_param']['key_definitive_3']
        definitive_3_attribute = config['definitive_3_attribute']
        defi_3_df = thirdparty_data.get_thirdparty_data(state,s3, bucket, key_definitive_3, definitive_3_attribute, config,datapulllogger)
        
        ###***********Before concat start to change customer name columns to unify them
        defi_1_df.rename(columns={'Assisted Living Facility Name': 'name'}, inplace=True)
        defi_2_df.rename(columns={'Skilled Nursing Facility': 'name'}, inplace=True)
        defi_3_df.rename(columns={'Hospital Name': 'name'}, inplace=True)
        ###***********Concat_different CSV file for definitive data first before matching************
        defi_concat_df = pd.concat([defi_1_df,defi_2_df,defi_3_df],ignore_index=True,sort=False)
        
        #********read in assitant living facility
        key_str_1 = config['Third_party_S3_param']['key_str_1']
        str_1_attribute = config['str_1_attribute']
        str_1_df = thirdparty_data.get_thirdparty_data(state,s3, bucket, key_str_1, str_1_attribute, config,datapulllogger)
        
        #********read in assitant living facility
        key_str_2 = config['Third_party_S3_param']['key_str_2']
        str_2_attribute = config['str_2_attribute']
        str_2_df = thirdparty_data.get_thirdparty_data(state,s3, bucket, key_str_2, str_2_attribute, config,datapulllogger)
        
        #********read in assitant living facility
        key_str_3 = config['Third_party_S3_param']['key_str_3']
        str_3_attribute = config['str_3_attribute']
        str_3_df = thirdparty_data.get_thirdparty_data(state,s3, bucket, key_str_3, str_3_attribute, config,datapulllogger)
        
        #********read in assitant living facility
        key_str_4 = config['Third_party_S3_param']['key_str_4']
        str_4_attribute = config['str_4_attribute']
        str_4_df = thirdparty_data.get_thirdparty_data(state,s3, bucket, key_str_4, str_4_attribute, config,datapulllogger)
        
        ###### **********Before concat, column name for management_phone and phone
        str_1_df.rename(columns={'Telephone':'Phone'},inplace=True)
        str_2_df.rename(columns={'Management Phone':'Phone'},inplace=True)
        str_3_df.rename(columns={'Management Phone':'Phone'},inplace=True)
        str_4_df.rename(columns={'Management Phone':'Phone'},inplace=True)
        
        ###***********Concat_different CSV file first before matching************
        str_concat_df = pd.concat([str_1_df,str_2_df,str_3_df,str_4_df],ignore_index=True,sort=False)
        str_concat_df['Latitude'] = str_concat_df['Latitude'].fillna(0)
        str_concat_df['Longitude'] = str_concat_df['Longitude'].fillna(0)
        str_concat_df.rename(columns={'Hotel Name': 'name'}, inplace=True)
        
        ###  ********************************************************************************************
        ### ### e. define clean_df():********************************************************************
        ### *********************************************************************************************
        spec = importlib.util.spec_from_file_location("data_clean", "./data_clean_ML.py")
        data_clean = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_clean)
        
        clean_safegraph_df = data_clean.clean_df(safegraph_df, state, config,datacleanlogger)
        # End with a log message
        if  clean_safegraph_df.shape[0] > 0:
            datacleanlogger.info("Data Clean completed successfully for %s" % state)
        else:
            datacleanlogger.error("Data Clean is failed for %s" % state)
            
        #write a message to datacleanlog
        # ****************Read sysco data ****************
        # Load function to pull data from sysco database
        mod = imp.load_source("data_clean_Yelp","./data_clean_ML_Yelp.py")
        import data_clean_Yelp
        clean_yelp_df = data_clean_Yelp.clean_df(yelp_df, state, config,datacleanlogger)
        
        # End with a log message
        if  clean_yelp_df.shape[0] > 0:
            datacleanlogger.info("Data Clean completed successfully for %s" % state)
        else:
            datacleanlogger.error("Data Clean is failed for %s" % state)
            
        #write a message to datacleanlog
        # ****************Read sysco data ****************
        # Load function to pull data from sysco database
        mod = imp.load_source("data_clean_chain","./data_clean_ML_chainstore.py")
        import data_clean_chain
        clean_chain_df = data_clean_chain.clean_df(chain_df, state, config,datacleanlogger)
        
        # End with a log message
        if  clean_chain_df.shape[0] > 0:
            datacleanlogger.info("Data Clean completed successfully for %s" % state)
        else:
            datacleanlogger.error("Data Clean is failed for %s" % state)    
            
        #write a message to datacleanlog
        # ****************Read sysco data ****************
        # Load function to pull data from sysco database
        mod = imp.load_source("data_clean_defi","./data_clean_ML_definitive.py")
        import data_clean_defi
        clean_defi_df = data_clean_defi.clean_df(defi_concat_df, state, config,datacleanlogger)
        # End with a log message
        if  clean_defi_df.shape[0] > 0:
            datacleanlogger.info("Data Clean completed successfully for %s" % state)
        else:
            datacleanlogger.error("Data Clean is failed for %s" % state)
        
        #write a message to datacleanlog
        # ****************Read sysco data ****************
        # Load function to pull data from sysco database
        mod = imp.load_source("data_clean_str","./data_clean_ML_str.py")
        import data_clean_str
        clean_str_df = data_clean_str.clean_df(str_concat_df, state, config,datacleanlogger)
        
        # End with a log message
        if  clean_str_df.shape[0] > 0:
            datacleanlogger.info("Data Clean completed successfully for %s" % state)
        else:
            datacleanlogger.error("Data Clean is failed for %s" % state)
        #####*******************Concat defi_df, chain_df, yelp_df, and str together*****************
        third_df = pd.concat([clean_yelp_df,clean_chain_df,clean_defi_df,clean_str_df],ignore_index=True,sort=False)
        third_df.rename(columns={'id': "yelp_id", 'Company ID': 'chainstore_ID','STR Number':'STR ID'},inplace=True)
        third_df['CLEANSED_PHONE'] = third_df['CLEANSED_PHONE'].fillna(0)
        
        ###  ********************************************************************************************
        ### f. Define run_record_clustering():***********************************
        ### ********************************************************************
        # Start a log message
        masterlogger.info("Record Linkage Modelling Process Started for {}.....".format(state))
        
        spec = importlib.util.spec_from_file_location("main_record_linkage", "./ML_main_record_linkage.py")
        main_record_linkage = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_record_linkage)
        
        if (len(clean_safegraph_df) > 0) and (len(third_df) > 0):
            merged_cluster_df = main_record_linkage.main_record_linkage(clean_safegraph_df,third_df,state,config,algorithmlogger,outFolderPath,outfoldername)
            
            #*********************** End with a log message****************************************************** 
            if (len(merged_cluster_df) > 0):
                algorithmlogger.info("Matching Process completed successfully, and match pair number is {0}" .format(len(merged_cluster_df)))
            else:
                algorithmlogger.error("Modelling Process failed")
           
        else:
            merged_cluster_df = pd.DataFrame(data=None)

        # Start a log message
        masterlogger.info("Second Record Linkage Modelling Process Started in Unclustered_data Only for {}.....".format(state))
        
        # ****************Read sysco data ****************
        # Load function to pull data from sysco database
        spec = importlib.util.spec_from_file_location("second_record_linkage", "./ML_second_record_linkage.py")
        second_record_linkage = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(second_record_linkage)
        
        ############################Here is the process to update the unclustered data with random address number comparison*******************
        ###########################Hopefully more clusters will be generated here####################################################
        if (len(merged_cluster_df) != len(third_df) and ((len(clean_safegraph_df) > 0) and (len(third_df) > 0))):
            merged_cluster_df_2,merged_Uncluster_df_2_final = second_record_linkage.main_record_linkage(clean_safegraph_df,third_df,merged_cluster_df, state, config,algorithmlogger,outFolderPath,outfoldername)
            if (len(merged_cluster_df_2) > 0 or len(merged_Uncluster_df_2_final) > 0):
                algorithmlogger.info("Matching Process completed successfully, and match pair number is {0}" .format(len(merged_cluster_df_2)))
                algorithmlogger.info("Unmatch pair number is {0}" .format(len(merged_Uncluster_df_2_final)))
            else:
                algorithmlogger.error("Modelling Process failed")
        elif len(merged_cluster_df) == len(third_df):
            merged_cluster_df_2 = merged_cluster_df.copy()
            uncluster1_df_final = clean_safegraph_df[~clean_safegraph_df.safegraph_place_id.isin (merged_cluster_df['safegraph_place_id'])].reset_index()
            uncluster1_df_final['third_party_match_probs']=0.4        
            uncluster2_df = pd.DataFrame(data=None, columns=third_df.columns)
            merged_Uncluster_df_2 = pd.concat([uncluster1_df_final,uncluster2_df], axis=1)
            merged_Uncluster_df_2.drop('index', axis=1,inplace=True)
            merged_Uncluster_df_2_final = merged_Uncluster_df_2[['safegraph_place_id','yelp_id','chainstore_ID','Definitive ID','STR ID']]
        
        else: 
            merged_cluster_df_2 = pd.DataFrame(data=None)
            if len(third_df) == 0:
                uncluster1_df_final = clean_safegraph_df.copy()
                uncluster1_df_final['third_party_match_probs']=0.4        
                uncluster2_df = pd.DataFrame(data=None, columns=third_df.columns)
                merged_Uncluster_df_2 = pd.concat([uncluster1_df_final,uncluster2_df], axis=1)
                merged_Uncluster_df_2_final = merged_Uncluster_df_2[['safegraph_place_id','yelp_id','chainstore_ID','Definitive ID','STR ID']]
            else:
                uncluster1_df_final = pd.DataFrame(data=None, columns=clean_safegraph_df.columns)
                uncluster1_df_final['third_party_match_probs']=0.4   
                uncluster2_df = third_df.copy()
                merged_Uncluster_df_2 = pd.concat([uncluster1_df_final,uncluster2_df], axis=1)
                merged_Uncluster_df_2_final = merged_Uncluster_df_2[['safegraph_place_id','yelp_id','chainstore_ID','Definitive ID','STR ID']]
        ###  ********************************************************************************************
        ### g. Move all results back to S3:***********************************
        ### ********************************************************************
        # Start a log message
        masterlogger.info("Move all results back to S3 for {}.....".format(state))
        
        spec = importlib.util.spec_from_file_location("upload_2_s3", "./upload_2_s3.py")
        upload_2_s3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upload_2_s3)
        
        upload_status = upload_2_s3.upload_file(merged_cluster_df_2,merged_Uncluster_df_2_final,state,outFolderPath,outfoldername,logsFolderPath, logfoldername,config,outputlogger)

        if upload_status:
            outputlogger.info("File and log upload Process completed successfully")
        else:
            outputlogger.error("File and log upload Process completed successfully")
    ##################################End of for Loops#############################
    ##################################################################################################
    # ************************************ End of run handling ****************************************
    # ************************************ This is way to properly close all opening files ****************************************
    for handler in handlerlist:
        handler.close() 
    # ********************************** # Close all open log file handlers ***************************************************     
#    
# ************************Run the py script only under Main condition********************************
if __name__ == "__main__":

    main()