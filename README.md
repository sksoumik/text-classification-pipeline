# text-classification-pipeline
Text data clean, pre-process, augmentation, apply State-of-the-art NLP models
Here, I have used a simple wrapper called [simpletransformers](https://github.com/ThilinaRajapakse/simpletransformers), on top of the original transformers by huggingface.


## Installation
Using anaconda distribution: 

1. Configure the environment
```

conda create --name simpletransformer python=3.6.9
conda activate simpletransformer
pip install torch===1.2.0 torchvision===0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==2.11.0
pip install simpletransformers==0.41.1
git clone --recursive https://github.com/NVIDIA/apex.git
cd apex && pip install .
python
>>> import nltk
>>> nltk.download('stopwords')
>>> nltk.download('punkt')
>>> nltk.download('averaged_perceptron_tagger')
>>> exit()

```
clone this repo:
```
$ git clone git@github.com:sksoumik/text-classification-pipeline.git
$ cd text-classification-pipeline
```

## Run the project

#### Clean texts
(if needed)
```
$ python preprocessing/preprocess.py
```

#### Data augmentatiom
(if needed)
```
$ python augmentation/data_augmentation.py
```

#### Process data

```
$ python conditionals/prepare_data.py
```

#### Train model

```
$ python train/train_model.py

optional arguments

$ python train/train_model.py --num_train_epochs=15 --learning_rate=1e-5 --train_batch_size=4 --eval_batch_size=4 --base_model=bert --tokenizer=bert-base-cased  

```

