import pandas as pd
import time
import logging

""" @description
     read_csv_to_parquet
     Read a data file from the UCI datasets and store it into a parquet file.
    @parameter url - The URL of the CSV
    @parameter names - Column names
    @parameter store_path - The local path for the Parquet file
    @parameter timeit - True if timing operations
"""
def read_csv_into_parquet(url, names, store_path, timeit=True):
    try:
        if timeit: start=time.time()
        df = pd.read_csv(url, names=names)
        if timeit: logging.info('URL Read Time = {} seconds'.format(time.time()-start))
    except FileNotFoundError as fnf_exception:
        logging.error(str(fnf_exception))
    except Exception as unclassified_exception:
        logging.error('Unclassified exception\n{}'.format(str(unclassified_exception)))
    try:
        if timeit: start = time.time()
        df.to_parquet(store_path, compression='gzip')
        if timeit: logging.info('Parquet Creation Time = {} seconds'.format(time.time()-start))
    except TypeError as t_exception:
        logging.error(str(t_exception))
    except Exception as unclassified_exception:
        logging.error('Unclassified exception\n{}'.format(str(unclassified_exception)))


""" @description
     read_parquet_to_df
     Read data from a local pickle file into a data frame
    @parameter store_path - The local path for the Parquet file
    @parameter timeit - True if timing operations
"""
def read_parquet_to_df(store_path, timeit=True):
    try:
        # Read the data from the Pickle file
        start = time.time()
        df = pd.read_parquet(store_path)
        if timeit: logging.info('Parquet Read Time = {} seconds'.format(time.time()-start))
        return df
    except FileNotFoundError as fnf_exception:
        logging.error(str(fnf_exception))
    except Exception as unclassified_exception:
        logging.error('Unclassified exception\n{}'.format(str(unclassified_exception)))
    return None

