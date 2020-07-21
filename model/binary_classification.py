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



# split the dataframe into train and test
def split_data(data):
    # split the dataframe into train and test
    train_df = data.sample(frac=0.8,random_state=200)
    test_df = data.drop(train_df.index)
    # save the test data to disk for further analysis
    test_df.to_csv("data/test_df.csv", index=False)
    return train_df, test_df

# For binary classification
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
    train_df, test_df = split_data(data)
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
    
    # list true labels
    true_label = list(test_df['class_id'])
    
    # list all the texts from test data
    test_df_comments = []
    for i in test_df['comment']:
        test_df_comments.append(i) 
    
    # Make predictions on test data
    predictions, raw_outputs = model.predict(test_df_comments)

    # make a new dataframe that includes text, true label and predicted label
    final_evaluation_df = pd.DataFrame(
        {'text': test_df_comments,
        'true labels': true_label,
        'predicted labels': predictions
        })

    # save the evaluation sheet
    final_evaluation_df.to_csv("data/final_evaluation_df.csv", index=False)


if __name__ == "__main__":
    main()

