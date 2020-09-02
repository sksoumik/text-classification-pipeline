import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from simpletransformers.classification import ClassificationModel, ClassificationArgs
import logging
import sklearn
import argparse

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
# ignore warnings
warnings.filterwarnings("ignore")
# force output to display the full description of pandas
pd.set_option('display.max_colwidth', -1)

from plot_confusion_matrix import make_confusion_matrix



'''
The below function might be useful if the data is not already splitted in 
train and test set 
'''
# def data_split(data):
#     train_df, test_df = train_test_split(data, stratify=data['class_id'], 
#                                     test_size=0.20)
    
#     print(f"\nclass distribution in train set: \n{train_df['class_id'].value_counts()}")
#     print(f"\nclass distribution in test set: \n{test_df['class_id'].value_counts()}")
#     return train_df, test_df 


"""
read train and test data from csv file that was generated from
conditionals/prepare_data.py file. 

expecting no header, two columns:
first column contains the true label in texts; e.g. positive, negative, neutral etc.
second column contains the texts/ feauture 
"""


def read_data():
    train_path = input("Input train data path:  ")
    train_df = pd.read_csv(train_path, keep_default_na=False)
    test_path = input("Input test data path:  ")
    test_df = pd.read_csv(test_path, keep_default_na=False)
    return train_df, test_df




def training_data():
    train_data, test_data = read_data()
    class_len, label_details = class_length(train_data)

    train_data = train_data.filter(items=['feature', 'class_id'])
    test_data = test_data.filter(items=['feature', 'class_id'])
    # train_data = pd.DataFrame(train_data, header=None)
    # test_data = pd.DataFrame(test_data, header=None)
    print(f"train set sample: \n{train_data.head()}")
    return train_data, test_data, class_len, label_details 


def class_length(train_data):
    categories = train_data["class"].unique()
    category_ids = train_data["class_id"].unique()
    categories = list(categories)
    category_ids = list(category_ids)
    label_details = list(
        map(lambda x, y: x + ':' + str(y), categories, category_ids))
    print(label_details)
    length = len(categories)
    print(f"total number of classes: \n{length}")
    return length, label_details


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-ep",
                        "--num_train_epochs",
                        type=int,
                        default=15,
                        help="define number of epochs")

    parser.add_argument("-lr",
                        "--learning_rate",
                        type=float,
                        default=1e-5,
                        help="define the learning rate")

    parser.add_argument("-tbs",
                        "--train_batch_size",
                        type=int,
                        default=1,
                        help="define batch size of training")

    parser.add_argument("-ebs",
                        "--eval_batch_size",
                        type=int,
                        default=1,
                        help="define batch size of evaluation")

    parser.add_argument("-bm",
                        "--base_model",
                        type=str,
                        default="bert",
                        help="define the base model like xlnet/bert")

    parser.add_argument("-tk",
                        "--tokenizer",
                        type=str,
                        default="bert-base-cased",
                        help="define the tokenizer")

    args = vars(parser.parse_args())

    # read the data
    train_df, test_df, class_len, label_details = training_data()

    model_args = ClassificationArgs(num_train_epochs=args['num_train_epochs'],
                                    train_batch_size=args['train_batch_size'],
                                    eval_batch_size=args['eval_batch_size'])

    model = ClassificationModel(args['base_model'],
                                args['tokenizer'],
                                num_labels=class_len,
                                args=model_args)

    model.train_model(train_df)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(
        test_df, acc=sklearn.metrics.accuracy_score)
    # print the evaluation result
    print(f"Evaluation: \n{result}")
    '''
    Make evaluation report for test set
    feature, true_label, predicted label 
    '''
    true_label = list(test_df["class_id"])

    test_df_features = []
    for i in test_df["feature"]:
        test_df_features.append(i)

    # evaluate the model again for test set
    predictions, raw_outputs = model.predict(test_df_features)

    final_evaluation_df = pd.DataFrame({
        "feature": test_df_features,
        "true labels": true_label,
        "predicted labels": predictions,
    })
    # save the evaluation sheet to disk
    final_evaluation_df.to_csv("data/evaluation_sheet.csv", index=False)

    eval_data = pd.read_csv("data/evaluation_sheet.csv")

    true_label = list(eval_data["true labels"])
    predicted_labels = list(eval_data["predicted labels"])

    y_test = true_label
    y_pred = predicted_labels

    matrix = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix: \n{matrix}")

    labels = ["True Neg", "False Pos", "False Neg", "True Pos"]
    label_categories = label_details
    make_confusion_matrix(matrix,
                          group_names=labels,
                          categories=label_categories,
                          cmap="binary")

    # classification report

    y_true = true_label
    y_pred = predicted_labels
    target_names = label_categories
    report = classification_report(y_true, y_pred, target_names=target_names)

    # save the classification report as a txt file

    print(report)

    original_stdout = sys.stdout

    with open("data/classification_report.txt", "w") as f:
        sys.stdout = f
        print(report)
        sys.stdout = original_stdout


if __name__ == "__main__":
    train()
