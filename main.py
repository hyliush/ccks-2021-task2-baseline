from config import Config
from datetime import datetime
import os
from tqdm import tqdm
from load_data import load_tianchiner
import torch
import torch.optim as optim
from pipeline import Trainer,Tester
from fastNLP.core.metrics import SpanFPreRecMetric,AccuracyMetric

from torch.optim.lr_scheduler import LambdaLR
from fastNLP import logger
from fastNLP.core.batch import DataSetIter
from fastNLP.core.sampler import SequentialSampler,RandomSampler
import sys
import numpy as np
from fastNLP.embeddings import BertEmbedding
from model import BiLSTMCRF

debug = False
colab = False
if colab:
    sys.argv =['name',
            '--model_name_or_path', '../pretrained_model/chinese_roberta_wwm_ext_pytorch',
            '--data_dir', '../data/tianchi',
            #'--do_test',
            '--do_train',
            '--save_logs',
            '--n_epochs', '3',
            '--num_classification', '3',
            '--max_seq_length', '80',
            '--train_batch_size', '16',
            '--eval_batch_size', '16',
            '--task', '0', '1',
            '--early_stop', '6',
            '--update_every', '2',
            '--validate_every', '1000',
            '--print_every', '4',
            '--warmup_steps', '0',
            '--warmup_proportion', '0.1',
            '--learning_rate', '2e-5',
            '--adam_epsilon', '1e-6',
            '--weight_decay', '1e-4']

    parser = Config.get_parser()
    args = parser.parse_args()
else:
    args = Config.get_default_cofig()

if debug:
    args.save_runs = True
    args.save_logs = True
    args.validate_every = 50

print(args)

cache_dir = os.path.join(args.data_dir,'cache')
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

output_dir = os.path.join(args.data_dir,'outputs')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if args.save_logs:
    # logs_dir
    logs_dir = os.path.join(output_dir,'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    logger.add_file(os.path.join(logs_dir,'{}'.format(str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))),level='info')


# prepare data
train_cache_name = os.path.join(cache_dir,'data_cache{}_{}'.format(args.train_batch_size,args.eval_batch_size))
pretrain_unigram_path = 'D:/NLP_competition/NER_data/pretrain/chinese/gigaword_chn.all.a2b.uni.ite50.vec'
pretrain_bigram_path = 'D:/NLP_competition/NER_data/pretrain/chinese/gigaword_chn.all.a2b.bi.ite50.vec'

datasets, vocabs, embeddings = load_tianchiner(args.data_dir, pretrain_unigram_path,
                                               pretrain_bigram_path,
                                               _refresh=False, index_token=False,
                                               _cache_fp=train_cache_name,
                                               char_min_freq=args.char_min_freq,
                                               bigram_min_freq=args.bigram_min_freq,
                                               only_train_min_freq=args.only_train_min_freq)
train_dataloader = DataSetIter(dataset=datasets['train'], batch_size=args.train_batch_size, sampler=RandomSampler())
dev_dataloader = DataSetIter(dataset=datasets['dev'], batch_size=args.eval_batch_size, sampler=SequentialSampler())
test_dataloader = DataSetIter(dataset=datasets['test'], batch_size=args.eval_batch_size, sampler=SequentialSampler())

# print(embeddings['char'].embedding.weight[:10])
def norm_static_embedding(x,norm=1):
    with torch.no_grad():
        x.embedding.weight /= (torch.norm(x.embedding.weight, dim=1, keepdim=True) + 1e-12)
        x.embedding.weight *= norm

print('embedding:{}'.format(embeddings['char'].embedding.weight.size()))
print('norm embedding')
for k,v in embeddings.items():
    norm_static_embedding(v,args.norm_embed)

# prepare model
bert_embedding = BertEmbedding(vocabs['char'], model_dir_or_name=args.model_name_or_path, requires_grad=False,
                                       word_dropout=0.01)
model = BiLSTMCRF(char_embed = embeddings['char'], num_classes=len(vocabs['label']), num_layers=1,
                      hidden_size=args.lstm_hidden_size, dropout=0.5,init=args.init,
                      bigram_embed=embeddings['bigram'],bert_embed=bert_embedding)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# optimizer
bert_embedding_param = list(model.bert_embedding.parameters())
bert_embedding_param_ids = list(map(id, bert_embedding_param))

bigram_embedding_param = list(model.bigram_embed.parameters())
char_embedding_param = list(model.char_embed.parameters())
embedding_param = bigram_embedding_param + char_embedding_param

embedding_param_ids = list(map(id, embedding_param))
non_embedding_param = list(filter(
    lambda x: id(x) not in embedding_param_ids and id(x) not in bert_embedding_param_ids,
    model.parameters()))
param_ = [{'params': non_embedding_param}, {'params': embedding_param, 'lr': args.learning_rate},
          {'params': bert_embedding_param, 'lr': 0.05 * args.learning_rate}]

optimizer = optim.SGD(param_,lr=args.learning_rate,momentum=args.momentum,
                      weight_decay=args.weight_decay)

# prepare metrics
encoding_type = 'bioes'
f1_metric = SpanFPreRecMetric(tag_vocab=vocabs['label'],pred='pred',target='target',seq_len='seq_len',encoding_type=encoding_type, f_type = 'macro')
acc_metric = AccuracyMetric(pred='pred',target='target',seq_len='seq_len',)
acc_metric.set_metric_name('label_acc')
metrics = [
    f1_metric,
    acc_metric
]

# prepare scheduler epoch end
scheduler = LambdaLR(optimizer, lambda ep: 1 / (1 + 0.05*ep))

# callback
class Unfreeze_Callback():
    def __init__(self, bert_embedding, fix_epoch_num):
        self.bert_embedding = bert_embedding
        self.fix_epoch_num = fix_epoch_num
        assert self.bert_embedding.requires_grad == False

    def on_epoch_begin(self, epoch):
        if epoch == self.fix_epoch_num + 1:
            self.bert_embedding.requires_grad = True

call_back = Unfreeze_Callback(model.bert_embedding,args.fix_bert_epoch)

def predict_train_dev(predictor,dataset_name='dev'):
    print('==================predicting {}==================='.format(dataset_name))
    pred_file = dataset_name
    if dataset_name == 'train':
        dataloader = train_dataloader
    if dataset_name == 'dev':
        dataloader = dev_dataloader
    if dataset_name == 'test':
        dataloader = test_dataloader

    dev_out = predictor.predict(dataloader)  # 预测结果
    dev_label_list = dev_out['infer_labels']
    dev_raw_char = datasets[dataset_name]['raw_chars']  # 原始文字
    dev_raw_label = datasets[dataset_name]['target']    # 标签

    dev_pred_text = open(os.path.join(output_dir,'{}.conll'.format(pred_file)), 'w', encoding='utf8')
    for _dev_char, _dev_label, _dev_label_pred in tqdm(zip(dev_raw_char, dev_raw_label, dev_label_list),
                                                       total=len(dev_label_list)):
        _dev_label_pred = np.reshape(_dev_label_pred, -1)
        for i, j, z in zip(_dev_char, _dev_label, _dev_label_pred):
            dev_pred_text.write(' '.join([str(i), str(j), str(z), tag_dict.get(j), tag_dict.get(z) + '\n']))
        dev_pred_text.write('\n')

tag_dict = vocabs['label'].idx2word
def predict_testset(predictor):
    # test 预测
    print('==================predicting test===================')
    test_out = predictor.predict(datasets['test'])  # 预测结果o
    test_label_list = test_out['pred']
    submit_test = open(os.path.join(output_dir,'hysh_addr_parsing_runid.txt'), 'w', encoding='utf8')
    final_test = open(r'D:\NLP_competition\NER_data\corpus\sequence_labelling\chinese_ner\TianchiNER\orignal_data\final_test.txt', 'r', encoding='utf8')
    for idx, (test_label_list_i, line) in tqdm(enumerate(zip(test_label_list, final_test.readlines())),
                                               total=len(test_label_list)):
        test_label_list_i = np.reshape(test_label_list_i, (1, -1))
        tag = ''
        for char in test_label_list_i[0]:
            tag_char = tag_dict.get(char)
            tag += tag_char + ' '
        if idx == len(test_label_list) - 1:
            new_line = line[:-1] + '\x01' + tag[:-1]
        else:
            new_line = line[:-1] + '\x01' + tag[:-1] + '\n'
        _ = submit_test.write(new_line)
    submit_test.close()
    final_test.close()


if args.do_train:
    load_model = False
    if load_model:
        model_path = os.path.join(output_dir, 'best_BertForSequenceClassification_2021-06-18-17-09-32_')
        if not os.path.exists(model_path):
            raise FileNotFoundError("folder `{}` does not exist. Please make sure model are there.".format(model_path))
        states = torch.load(model_path).state_dict()
        model.load_state_dict(states)

    # trainer
    trainer = Trainer(train_dataloader, model, optimizer, scheduler=None,
                update_every=args.update_every,n_epochs=args.n_epochs,
                 print_every=args.print_every,early_stop=args.early_stop,metrics=metrics,
                 dev_dataloader=dev_dataloader, validate_every=args.validate_every,
                save_path=output_dir,customize_model_name = None,seed=args.seed,debug=debug,
                writer=None,call_back=call_back)

    trainer.train()

    predictor = Tester(model)
    # train 预测
    predict_train_dev(predictor, 'train')
    # dev 预测
    predict_train_dev(predictor,'dev')
    # test 预测
    predict_testset(predictor)

if args.do_test:
    model_path = os.path.join(output_dir,'best_BiLSTMCRF_f_2021-06-16-20-08-36')
    states = torch.load(model_path).state_dict()
    model.load_state_dict(states)
    model = model.to(device)
    predictor = Tester(model)
    # train 预测
    predict_train_dev(predictor, 'train')

    # dev 预测
    predict_train_dev(predictor,'dev')
    # test 预测
    predict_testset(predictor)
