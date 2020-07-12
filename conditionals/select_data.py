import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
# ignore warnings
warnings.filterwarnings("ignore")
# force output to display the full description of pandas
pd.set_option('display.max_colwidth', -1)


def select_primary_condition(data, classification_or_item,
                             name_of_the_classification_or_item,
                             phenomenon_value):
    if classification_or_item == 'classification':
        # create a new dataframe based on the classification
        data = data[data['classification'] ==
                    name_of_the_classification_or_item]
    else:
        data = data[data['item'] == name_of_the_classification_or_item]

    data.loc[data['phenomenon'] == phenomenon_value, 'class_id'] = '1'
    data.loc[data['phenomenon'] != phenomenon_value, 'class_id'] = '0'

    data.to_csv('data/' + classification_or_item + "_" +
                name_of_the_classification_or_item + "_" + phenomenon_value +
                ".csv",
                index=False)

    return data


def main():
    data = pd.read_csv("data/clean_data.csv")
    input_condition = input(
        "Enter classification or item as the primary condition; e.g. item/classification: ")
    name_of_the_classification_or_item = input(
        "Enter the name of the classification or item; e.g. Cup holder/Handling and Steering: ")
    phenomenon_value = input("Enter the phenomenon value name; e.g. LE  Location: ")
    select_primary_condition(data, input_condition,
                             name_of_the_classification_or_item,
                             phenomenon_value)
    print("INFO: DONE! new data saved in the data/ directory.")


if __name__ == "__main__":
    main()
