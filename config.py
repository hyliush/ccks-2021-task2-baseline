import argparse
import sys
import logging
# MODEL_CLASSES = {'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)}
# ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in ( BertConfig,)), ())

class Config(object):
    @classmethod
    def get_parser(cls):
        parser = argparse.ArgumentParser()
        ## Required parameters
        parser.add_argument("--early_stop",default = 10,type = int)
        parser.add_argument("--print_every",default = 5,type = int)
                            
        parser.add_argument('--update_every', type=int, default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument('--validate_every', type=int, default=-1)

        parser.add_argument("--data_dir", default=None, type=str, required=True,
                            help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
        parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                            help="Path to pre-trained model or shortcut name selected in the list: ")

        parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
        parser.add_argument("--do_test", action='store_true', help="Whether to run testing.")
        parser.add_argument("--save_logs", action='store_true', help="Whether to save logs.")
        parser.add_argument("--save_runs", action='store_true', help="Whether to save writer .")

        parser.add_argument('--num_classification',required=True,type=int,default=3,
                            help="the number of categories")
        parser.add_argument("--n_epochs", default=100, type=int,
                            help="Total number of training epochs to perform.")
        parser.add_argument('--predict_text', default="", type=str, help="Predict sentiment on a given sentence")
        parser.add_argument('--predict_filename', default="", type=str, help="Predict sentiment on a given file")
        parser.add_argument("--max_seq_length", default=128, type=int,required=True,
                            help="The maximum total input sequence length after tokenization. Sequences longer "
                                 "than this will be truncated, sequences shorter will be padded.")
        parser.add_argument("--weight_decay", default=1e-4, type=float,
                            help="Weight deay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                            help="Epsilon for Adam optimizer.")
        parser.add_argument("--momentum", default=0.9, type=float,
                            help="momentum for optimizer.")
        parser.add_argument("--train_batch_size", default=8, type=int,
                            help="Batch size per GPU/CPU for training.")
        parser.add_argument("--learning_rate", default=2e-5, type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--warmup_steps", default=10, type=int,
                            help="Linear warmup over warmup steps.")
        parser.add_argument("--warmup_proportion", default=0.1, type=float,
                            help="Linear warmup over warmup_proportion.")
        parser.add_argument('--char_min_freq', default=1, type=int)
        parser.add_argument('--bigram_min_freq', default=1, type=int)
        parser.add_argument('--only_train_min_freq', default=True)
        parser.add_argument('--norm_embed', default=True)
        parser.add_argument("--lstm_hidden_size", default=160, type=int,help="")
        parser.add_argument('--init', default='uniform', help='norm|uniform')
        parser.add_argument('--fix_bert_epoch', type=int, default=20)

        ## Other parameters
        parser.add_argument('--task',type=int,nargs='+',
                            help="The task. for example dummy classification[0,1]")
        parser.add_argument("--config_name", default="", type=str,
                            help="Pretrained config name or path if not the same as model_name")
        parser.add_argument("--tokenizer_name", default="", type=str,
                            help="Pretrained tokenizer name or path if not the same as model_name")
        parser.add_argument("--eval_batch_size", default=8, type=int,
                            help="Batch size per GPU/CPU for evaluation.")
        parser.add_argument("--do_lower_case", action='store_true',
                            help="Set this flag if you are using an uncased model.")
        parser.add_argument("--eval_steps", default=-1, type=int,
                            help="")
        parser.add_argument("--lstm_layers", default=2, type=int,
                            help="")
        parser.add_argument("--lstm_dropout", default=0.5, type=float,
                            help="")
        parser.add_argument("--split_num", default=3, type=int,
                            help="text split")
        parser.add_argument('--logging_steps', type=int, default=50,
                            help="Log every X updates steps.")
        parser.add_argument('--save_steps', type=int, default=50,
                            help="Save checkpoint every X updates steps.")
        parser.add_argument('--overwrite_output_dir', action='store_true',
                            help="Overwrite the content of the output directory")
        parser.add_argument('--overwrite_cache', action='store_true',
                            help="Overwrite the cached training and evaluation sets")
        parser.add_argument('--seed', type=int, default=42,
                            help="random seed for initialization")
        return parser

    @classmethod
    def get_default_cofig(cls):
        sys.argv =['name',
                '--model_name_or_path', r'D:\NLP_competition\NER_data\pretrain\chinese\chinese_roberta_wwm_ext_pytorch',
                #'--data_dir', r'D:\NLP_competition\NER_data\corpus\classification\cu_data\data_StratifiedKFold_125/data_origin_0',
                #'--data_dir', r'D:\NLP_competition\NER_data\corpus\match\tianchi',
                '--data_dir', r'D:\NLP_competition\NER_data\corpus\sequence_labelling\chinese_ner\TianchiNER\orignal_data',
                #'--do_test',
                '--do_train',
                '--save_logs',
                '--n_epochs', '100',
                '--num_classification', '3',
                '--max_seq_length', '80',
                '--train_batch_size', '4',
                '--eval_batch_size', '8',
                '--fix_bert_epoch', '20',
                '--task', '0', '1',
                '--early_stop', '6',
                '--update_every', '2',
                '--validate_every', '-1',
                '--print_every', '4',
                '--warmup_steps', '0',
                '--warmup_proportion', '0.1',
                '--learning_rate', '1e-3',
                '--adam_epsilon', '1e-6',
                '--weight_decay', '1e-4']
        parser = cls.get_parser()
        args = parser.parse_args()
        return  args

    def __str__(self):
        for k, v in self.__dict__.items():
            print('{}:{}\n'.format(k, v))



if __name__ == '__main__':
    config = Config()
    args = config.get_default_cofig()


