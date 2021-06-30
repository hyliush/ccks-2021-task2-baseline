# ccks-2021-task2-baseline 90+

provide a baseline for ccks-2021-task2(address parsing)


## Simple way to run for ccks2021 task2

1. prepare bert_pretrained model and revised  '--model_name_or_path' 

2. prepare bigram and char embedding and revised pretrain_unigram_path,pretrain_bigram_path

3. prepare dataset and revised  '--data_dir'

4. run on colab (main.ipynb). turn debug on False before run main.

5. run on local device(main.py). turn debug on False before run main.

## Stucture
The main module contains the follow files:

- The load_data.py
Text process -> read a file and convert it to a format for model (fastNLP package).

- model.py
Build Model  ->  char,bigram and bert embedding + Bi-LSTM + CRF model and other model can be added. 

- pipeline.py contain two classes.  Trainer is for training process. Tester is for testing process which contains model predict and evaluation.

- main.py
main file to run on local device.

- config.py 

- data folder contains files(train.conll,dev.conll,test.conll).

## Other 
- 1.one can add some trick to import prediction performance. For example model average,Pseudo label,model stacking. Details can be seen[BDCI top1 scheme](https://github.com/cxy229/BDCI2019-SENTIMENT-CLASSIFICATION).
- 2.model_name_or_path contains pretrained bert model files (.bin,.json,.txt) which can be downloaded [Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm) for chinese text (also support other language).
- 3.char and bigram embedding  can be downloaded from [Flat](https://github.com/LeeSureman/Flat-Lattice-Transformer)
- 4.Flat model achieves 87.93, cross validate:88.73, pseudo-labelling: 89.73
- 5.char-bigram-bilstm 86.88
- 6.[biaffine-ner](https://github.com/amir-zeldes/biaffine-ner), 81.63
- 7.our provided model(char-bigram-bert-bilstm-crf) 90+

