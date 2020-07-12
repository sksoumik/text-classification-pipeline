# text-classification-pipeline
text data clean, pre-process, augmentation, apply SOTA models


## Installation

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

## Run The Project
1. python preprocessing/preprocess.py