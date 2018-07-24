# Import the Pickle library
import pickle
import pandas as pd
import time
import logging

""" @description
     read_csv_to_pickle
     Read a data file from the UCI datasets and store it into a pickle file.
    @parameter url - The URL of the CSV
    @parameter names - Column names
    @parameter store_path - The local path for the Pickle file
    @parameter timeit - True if timing operations
"""
def read_csv_into_pickle(url, names, store_path, timeit=True):
    try:
        if timeit: start=time.time()
        df = pd.read_csv(url, names=names)
        if timeit: logging.info('URL Read Time = {} seconds'.format(time.time()-start))
        if timeit: start = time.time()
        with open(store_path, 'wb') as handle:
            pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if timeit: logging.info('Pickle Creation Time = {} seconds'.format(time.time()-start))
    except TypeError as t_exception:
        logging.error(str(t_exception))
    except Exception as unclassified_exception:
        logging.error('Unclassified exception\n{}'.format(str(unclassified_exception)))


""" @description
     read_pickle_to_df
     Read data from a local pickle file into a data frame
    @parameter store_path - The local path for the Pickle file
    @parameter timeit - True if timing operations
"""
def read_pickle_to_df(store_path, timeit=True):
    try:
        # Read the data from the Pickle file
        start = time.time()
        with open(store_path, 'rb') as handle:
            df = pickle.load(handle)
        logging.info('Pickle Read Time = {} seconds'.format(time.time()-start))
        return df
    except FileNotFoundError as fnf_exception:
        logging.error(str(fnf_exception))
    except Exception as unclassified_exception:
        logging.error('Unclassified exception\n{}'.format(str(unclassified_exception)))
    return None


