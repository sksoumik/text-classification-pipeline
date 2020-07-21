"""
Text class to numeric ID mapping for multi-class classification

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
# ignore warnings
warnings.filterwarnings("ignore")
# force output to display the full description of pandas
pd.set_option('display.max_colwidth', -1)


"""
This function assumes data has multi classes

column_name: column contains the class names;
for example: positive, neutral, negative ... etc. 
"""

def class_to_ID(df):
    for i in range(len(df['column_name'].unique())):
        df.loc[df['column_name'] == df['column_name'].unique()[i], 'class_id'] = i
    # make sure class IDs are int64 format
    df['class_id'] = df['class_id'].astype('Int64')
    # print the class distribution
    print(f"\nClass distribution: \n{df['class_id'].value_counts()}")
    # save the new data to disk.
    df.to_csv('data/data_with_class_id.csv')
    return df


# undersample the dataset
def sampling_k_elements(group, k=class_data_points):
    class_data_points = int(
        input("Enter the amount of data that you want for each class: "))
    if len(group) < k:
        return group
    return group.sample(k)


def main():
    data = pd.read_csv("data/clean_data.csv")
    positive_class = input("What is the name of your positive class: ")
    negative_class = input("What is the name of your negative class: ")
    df = class_to_ID(data, positive_class, negative_class)
    # undersample the dataset
    balanced = df.groupby('class_id').apply(sampling_k_elements).reset_index(
        drop=True)
    print(
        f"\n Class distribution after class balancing: \n{balanced['class_id'].value_counts()}"
    )
    balanced.to_csv("data/class_balanced_data.csv", index=False)
    print("[INFO] DONE! new data saved in the data/ directory.")


if __name__ == "__main__":
    main()
