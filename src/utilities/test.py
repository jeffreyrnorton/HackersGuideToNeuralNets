import logging
import setup_logging

setup_logging.setup_logging()

from parquet_util import read_csv_into_parquet

# Read the CSV (at URL) into a Parquet file
read_csv_into_parquet(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'],
    store_path='iris.parquet'
)
