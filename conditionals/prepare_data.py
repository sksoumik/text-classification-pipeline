"""
Text class to numeric ID mapping for multi-class classification

"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
# ignore warnings
warnings.filterwarnings("ignore")
# force output to display the full description of pandas
pd.set_option('display.max_colwidth', -1)
'''
read train and test data from excel file
expecting no header, two columns:
first column contains the true label in texts; e.g. positive, negative, neutral etc.
second column contains the texts/ feauture 
'''


def read_data():
    train_path = input("Input train data path:  ")
    train_data = pd.read_excel(train_path, keep_default_na=False, header=None)
    # add a new column `train` for train data
    train_data["train-test"] = "train"
    test_path = input("Input test data path:  ")
    test_data = pd.read_excel(test_path, keep_default_na=False, header=None)
    # add a new column `test` for test data
    test_data["train-test"] = "test"
    # append train and test data together
    data = train_data.append(test_data, ignore_index=True)
    # rename columns
    rename_cols = {
        0: "class",
        1: "feature",
    }
    data = data.rename(columns=rename_cols)
    return data


'''
convert text classes to numeric value.
e.g. 
positive -> 0
negative -> 1
neutral -> 2 
etc.... 
'''


def encode_class(data):
    for i in range(len(data["class"].unique())):
        data.loc[data["class"] == data["class"].unique()[i], "class_id"] = i

    data["class_id"] = data["class_id"].astype("Int64")
    return data


'''
Undersample and balance all classes' data points to avoid biasness. 
make all class's data points equal 
'''


def sampling_train_data(group, k=int(
        input(
            "Enter the amount of data that you want for each class for train set: "
        ))):
   
    if len(group) < k:
        return group
    return group.sample(k)


def sampling_test_data(group, k=int(
        input(
            "Enter the amount of data that you want for each class for test set: "
        ))):

    if len(group) < k:
        return group
    return group.sample(k)


def balance_data(data):
    train_data = data[data["train-test"] == "train"]
    test_data = data[data["train-test"] == "test"]
    train_df = train_data[["feature", "class_id", "class"]]
    test_df = test_data[["feature", "class_id", "class"]]
    train_df = train_df.groupby('class_id').apply(
        sampling_train_data).reset_index(drop=True)
    test_df = test_df.groupby('class_id').apply(
        sampling_test_data).reset_index(drop=True)
    # save train and test data to disk
    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train_data.csv", index=False)
    test_df.to_csv("data/test_data.csv", index=False)


def main():
    data = read_data()
    data = encode_class(data)
    balance_data(data)


if __name__ == "__main__":
    main()
