# UniRPG

This is code for the EMNLP 2022 Paper "UniRPG: Unified Discrete Reasoning over Table and Text as Program Generation".

# Environment
torch==1.7.1</br>
transformers==3.3.0</br>
fastNLP==0.6.0</br>
allennlp==2.0.1</br>
spacy==2.0.1</br>

# Training and Inference
The folder UniRPG_full is for training UniRPG with the derivation annotations while the folder UniRPG_weak is for the setting without the derivation annotations.
## Preprocessing
You should run the following scripts to preprocess training/dev/test data.</br>
```bash scripts/prepare_data_train.sh```</br>
```bash scripts/prepare_data_dev.sh```</br>
```bash scripts/prepare_data_test.sh```</br>
## Training
You should run the following scripts to train the UniRPG</br>
```bash scripts/train_bart_large.sh```
## Validation
You should run the following scripts to evaluate the trained model in dev set</br>
```bash scripts/validate.sh```</br>
```bash scripts/eval.sh```
## Inference
You should first predict the programs and execute them to get the answers of instances in test set.</br>
```bash scripts/predict.sh ```</br>
```bash scripts/execute.sh```</br>
# Thanks for your Citation.


