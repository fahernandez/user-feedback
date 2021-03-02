from sagemaker import get_execution_role
import pandas as pd
import boto3

bucket_name = 's3://sagemaker-user-feedback/'
role = get_execution_role()

def save_file(result, key_name):
    result.to_csv('{}{}'.format(bucket_name, key_name))

def get_file(key_name, index=None):
    return pd.read_csv("{}{}".format(bucket_name, key_name), index_col=index if index is not None else None)

def get_comments():
    # get comments from s3
    data = pd.read_csv("{}proccesed_comments.csv".format(bucket_name), index_col='id')
    variables = data.drop('primary_category', axis = 1)
    response = data['primary_category'].values
    
    return data, variables, response