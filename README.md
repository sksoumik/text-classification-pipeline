# text-classification-pipeline
Text data clean, pre-process, augmentation, apply State-of-the-art NLP models
Here, I have used a simple wrapper called [simpletransformers](https://github.com/ThilinaRajapakse/simpletransformers), on top of the original transformers by huggingface.


#### clone this repo:
```
$ git clone git@github.com:sksoumik/text-classification-pipeline.git
$ cd text-classification-pipeline
```

#### Installation
Using anaconda distribution configure the environment

```
$ conda create --name simpletransformer python=3.6.9
$ conda activate simpletransformer
$ pip install torch===1.2.0 torchvision===0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install transformers==2.11.0
$ pip install simpletransformers==0.41.1
$ git clone --recursive https://github.com/NVIDIA/apex.git
$ cd apex && pip install .
$ python
>>> import nltk
>>> nltk.download('stopwords')
>>> nltk.download('punkt')
>>> nltk.download('averaged_perceptron_tagger')
>>> exit()
```
You can also use the requirements.txt for the required packages
```
$ pip install requirements.txt
```

## Run the project

#### Clean texts
(if needed)
```
$ python preprocessing/preprocess.py
```

#### Data augmentatiom

(if needed)

Input data must be in the following format:

labels, texts 

```
1   neil burger here succeeded in making the mystery of four decades back the springboard for a more immediate mystery in the present 
0   it is a visual rorschach test and i must have failed 
0   the only way to tolerate this insipid brutally clueless film might be with a large dose of painkillers
```


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
```
optional arguments

```
$ python train/train_model.py --num_train_epochs=15 --learning_rate=1e-5 --train_batch_size=4 --eval_batch_size=4 --base_model=bert --tokenizer=bert-base-cased  
```
