# UniRPG

This is code for the EMNLP 2022 Paper "UniRPG: Unified Discrete Reasoning over Table and Text as Program Generation".  Our code is based on the repository [TAT-QA](https://github.com/NExTplusplus/TAT-QA}).

# Environment
torch==1.7.1</br>
transformers==3.3.0</br>
fastNLP==0.6.0</br>
allennlp==2.0.1</br>
spacy==2.0.1</br>

# Training and Inference
The folder ```UniRPG_full``` is for training UniRPG with the derivation annotations while the folder ```UniRPG_weak``` is for the setting without the derivation annotations.
## Preprocessing
First download pre-trained model BART and put files in the folder ```plm```, then you should run the following scripts to preprocess training/dev/test data.</br>
```bash scripts/prepare_data_train.sh```</br>
```bash scripts/prepare_data_dev.sh```</br>
```bash scripts/prepare_data_test.sh```</br>
The preprocessed train/dev/test data is stored in the folder ```tag_op/cache/```

## Training
You should run the following scripts to train the UniRPG</br>
```bash scripts/train_bart_large.sh```</br>
The trained UniRPG model is saved in the folder ```checkpoint```
## Validation
First check the saved path of model in the following scripts and then run them to evaluate the trained model in dev set</br>
```bash scripts/validate.sh```</br>
```bash scripts/execute.sh```</br>
```bash scripts/eval.sh```
## Inference
Please check the saved path of model in the following scripts, and then predict the programs and execute them to get the answers of test instances.</br>
```bash scripts/predict.sh ```</br>
```bash scripts/execute.sh```</br>
# Thanks for your Citation.


