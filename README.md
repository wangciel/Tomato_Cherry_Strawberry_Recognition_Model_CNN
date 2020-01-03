## Student Name: YiFan Wang
## Student ID:300304266

## Folder structure and file containing:

<br> dir:
<br>/
<br>    /data
<br>        - /test
<br>          - /cherry
<br>          - /strawberry
<br>          - /tomato
<br>    /model
<br>        - model.h5
<br>    /plots
<br>        - some plots.png
<br>    /Train_data_imputation
<br>        - /cherry
<br>        - /strawberry
<br>        - /tomato
<br>    /util
<br>        - cnn_model_maker.py
<br>        - data_loader.py
<br>baseline_mlp.py
<br>test.py
<br>train.py
<br>ReadMe.txt


## Environment:
## common need:
numpy, sk-learn, matplotlib, imutils, opencv

for run test.py/ train.py you need Tensorflow 1.1, keras

A successful environment for GPU training:
Python3.6, CUDA 9, cudnn 7, Tensorflow 1.1


## How To Run My Program:
test.py run it on pycharm ide or

in commend line:
cd ~/your test.py folder
python test.py your_test_dataset_dir
*(dataset dir should have same structure as above)

train.py run it on pycharm ide and replace train_data_dir in util/data_loader.py
load_cnn_train_data(with_imputation=True) means dataset mixed imputation data vice versa.
