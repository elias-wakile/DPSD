# differently private spam filter using deep learning


## welcome to our project. this project tries to implement a differently private spam filter using deep learning

## installation

### windows

you can create a python environment, or you can simply run the command:

`pip install -r requirements.txt`

and you are set


### aquarium computers

1. create an virtual env:
   `virtualenv my-first-venv`
2. activate the env:
   `source my-first-venv/bin/activate.csh`
3. install requirements:
   `pip install -r requirements.txt`
4. activate cuda if you can:
   `module load cuda`

it is important to note that some aquarium computers have old gpus
which may not work well with newer versions of pytorch. if this is the case 
we recommend using the cpu to run our project.
if you want to force the project to work with gpu/cpu you may change the 
`device` variable in the file **_utils.py_** 
by default we chose cpu to work with as many devices as we can

for more information you may go to:[https://wiki.cs.huji.ac.il/wiki/Python](https://wiki.cs.huji.ac.il/wiki/Python)
   

## Running the code

###there are two ways to run our code. you can train a model of your own, or you can run our models on an email

#### training a model

**you will need to download the dataset file from** 
[https://drive.google.com/drive/folders/19sDQZ2OrCYXJpf1SN19hls8F-K5l87lp](https://drive.google.com/drive/folders/19sDQZ2OrCYXJpf1SN19hls8F-K5l87lp)

training a model is very simple. you will use the file training.py

`python training.py [-h] --model_type {LSTM,LSTM_ATTN} --privacy_training 
{NORMAL,TRAINING,BUDGET} [--max_grad_norm MAX_GRAD_NORM] [--delta DELTA]
[--eps EPS] [--noise_multiplier NOISE_MULTIPLIER]`

#### params:
1. --model_type: the type of model to train. LSTM or LSTM_ATTN
2. --privacy_training: the type privacy we want to train with:
   1. NORMAL - No privacy
   2. TRAINING - Normal training with DP-SGD. 
   3. BUDGET - Training with DP-SGD. model has (epsilon,delta) budget
3. --max_grad_norm: maximum gradient norm for training
4. --delta: delta used for training
5. --eps: epsilon used for training. only works for BUDGET mode
6. --noise_multiplier: variance for noise training. only works for TRAINING mode


### running models

running our models is very simple. you will use the file run_model.py

there are 10 available models.
you will use the following command:

`python run_model.py [-h] --subject SUBJECT --msg MSG --model_type {LSTM,
LSTM_ATTN}
--privacy_training {NORMAL,TRAINING,BUDGET} --accuracy_type
{BEST,PRIVATE}`


#### params:
1. --subject: email subject
2. --msg: email text
3. --model_type: the type of model to use. LSTM or LSTM_ATTN
4. --privacy_training: the type privacy of the model trained:
   1. NORMAL - No privacy
   2. TRAINING - Normal training with DP-SGD.
   3. BUDGET - Training with DP-SGD. model has (epsilon,delta) budget
5. --accuracy_type: type of accuracy we would like to use.
   1. BEST -- model with the highest accuracy
   2. PRIVATE -- model with the best privacy values. usually has lower accuracy
      (available when privacy_training != Normal)
