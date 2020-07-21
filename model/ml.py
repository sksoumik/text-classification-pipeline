"""
Binary classification: NLP

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel
import logging
import sklearn
import argparse

from plot_confusion_matrix import make_confusion_matrix

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
# ignore warnings
warnings.filterwarnings("ignore")
# force output to display the full description of pandas
pd.set_option('display.max_colwidth', -1)


"""
This function expects at least two columns with:
text: context
class_id: class ID generated from the conditionals/binary_class_map.py
"""
def read_data():
    data_path = input(
        "Input CSV file path that is created from conditionals: ")
    data = pd.read_csv(data_path)
    data = data.filter(items=['text', 'class_id'])
    print(f"class distribution:\n{data.class_id.value_counts()}")
    return data


# under-sample dataset to make all class balanced
def class_balancing(data):
    counts = data['class_id'].value_counts()
    # Undersample the dataset
    minority_class_length = len(data[data['class_id'] == max(counts.keys())])
    # print(f"\nINFO: minority class length: {minority_class_length}")
    majority_class_indices = data[data['class_id'] == min(counts.keys())].index
    # print(f"\nMajority class indices: \n{majority_class_indices}")
    random_majority_indices = np.random.choice(majority_class_indices,
                                               minority_class_length,
                                               replace=False)
    minority_class_indices = data[data['class_id'] == max(counts.keys())].index
    under_sample_indices = np.concatenate(
        [minority_class_indices, random_majority_indices])
    under_sample = data.loc[under_sample_indices]
    print(
        f"class distribution after under sampling: \n{under_sample.class_id.value_counts()}"
    )
    return under_sample


# split the dataframe into train and test
def split_data(data):
    train_df, test_df = train_test_split(data, test_size=0.2)
    return train_df, test_df


def create_model(base_model, weights, args):
    model = ClassificationModel(base_model, weights, args=args)
    return model


def main():
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

    parser.add_argument("-msl",
                        "--max_seq_length",
                        type=int,
                        default=100,
                        help="define maximum word sequence for a sentence")

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
                        default="xlnet",
                        help="define the base model like xlnet/bert")

    parser.add_argument(
        "-mw",
        "--model_weight",
        type=str,
        default="xlnet-base-cased",
        help=
        "define the base model weight like xlnet-base-cased/bert-base-uncased")

    args = vars(parser.parse_args())

    arguments = {
        'num_train_epochs': args['num_train_epochs'],
        'learning_rate': args['learning_rate'],
        'max_seq_length': args['max_seq_length'],
        'train_batch_size': args['train_batch_size'],
        'eval_batch_size': args['eval_batch_size'],
        'overwrite_output_dir': True,
    }

    data = read_data()
    balanced_data = class_balancing(data)
    train_df, test_df = split_data(balanced_data)
    # print(f"\nclass distribution in train data: \n{train_df.class_id.value_counts()}")
    # print(f"\nclass distribution in test data: \n{test_df.class_id.value_counts()}")

    # create the model
    model = create_model(args['base_model'],
                         args['model_weight'],
                         args=arguments)
    # train the model
    model.train_model(train_df)
    # evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(
        test_df, acc=sklearn.metrics.accuracy_score)
    print(f"Evaluation: \n{result}")

    confusion_matrix = np.array([[result['tn'], result['fp']],
                                 [result['fn'], result['tp']]])

    labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    categories = ['Zero', 'One']

    # plot the confusion matrix
    make_confusion_matrix(confusion_matrix,
                          group_names=labels,
                          categories=categories,
                          cmap='binary')

if __name__ == "__main__":
    main()

