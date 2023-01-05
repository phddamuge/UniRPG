# UniRPG

This is the code for the EMNLP 2022 Paper [UniRPG: Unified Discrete Reasoning over Table and Text as Program Generation](https://preview.aclanthology.org/emnlp-22-ingestion/2022.emnlp-main.508.pdf).  Our code is based on the repository [TAT-QA](https://github.com/NExTplusplus/TAT-QA).
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
The preprocessed train/dev/test data is stored in the folder ```tag_op/cache/```</br>
Under weak supervision setting, you should first run the following command to convert multi-span instances to count instances, and then run the above preprocessing scripts.</br>
```python3 tag_op/data/count_instance_construction.py```

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
# Thanks for your citation.
```
@inproceedings{zhou2022unirpg,
  doi = {10.48550/ARXIV.2210.08249},
  url = {https://arxiv.org/abs/2210.08249},
  author = {Zhou, Yongwei and Bao, Junwei and Duan, Chaoqun and Wu, Youzheng and He, Xiaodong and Zhao, Tiejun},
  title = {UniRPG: Unified Discrete Reasoning over Table and Text as Program Generation},
  publisher = {arXiv},
  year = {2022},
}

```

