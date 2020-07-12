"""
Select the data conditions

"""
import pandas as pd 

def select_condition(data, classification, phenomenon):
    # create a new dataframe based
    data = data[(data['classification'] == classification) & (data['phenomenon'] == phenomenon)]
    # add a new column for class id
    print(len(f"INFO: The length of the dataset {data}"))
    return data 



def class_map(data, phenomenon, ):
    unique_phenomenons = data.phenomenon.unique()



def main():
    pass

