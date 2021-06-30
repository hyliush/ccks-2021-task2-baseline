import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel,BertModel
import torch
from fastNLP.modules import ConditionalRandomField
from fastNLP.modules import LSTM
from collections import namedtuple
import torch.nn.functional as F
model_output=namedtuple('out',['loss','logits'])

class PointwiseMatching(BertPreTrainedModel):
    # 此处的 pretained_model 在本例中会被 ERNIE1.0 预训练模型初始化
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout( 0.1)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        #self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # module.allpy 迭代进入children，实际进入modules

        for n,m in self.named_children():
            if n=='classifier':
            # if m.__class__.__name__ == 'Linear':
            # if type(m) == nn.Linear
                if hasattr(self.config,'init'):
                    if self.config.init == 'normal':
                        nn.init.xavier_normal_(m.weight)
                    if self.config.init == 'uniform':
                        nn.init.xavier_uniform_(m.weight)
                else:
                    m.weight.fill_(1)
                m.bias.fill_(0)

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                add_graph=False,
                position_ids=None,
                labels=None,return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(input_ids, attention_mask,token_type_ids, position_ids)

        cls_embedding = outputs.get('pooler_output')
        #cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss=loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        else:
            loss = None

        if not return_dict or add_graph:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return {'loss':loss,'logits':logits}



class BiLSTMCRF(nn.Module):
    def __init__(self,char_embed,num_classes,num_layers,hidden_size,dropout,
                 bigram_embed=None, bert_embed=None,init=None):
        super().__init__()
        self.init = init
        self.char_embed = char_embed
        self.bigram_embed = bigram_embed
        self.bert_embedding = bert_embed

        self.use_bert = False
        self.use_bigram = False
        self.input_size = self.char_embed.embed_size
        if self.bigram_embed:
            self.use_bigram = True
            self.input_size += self.bigram_embed.embed_size

        if self.bert_embedding:
            self.use_bert = True
            self.input_size += self.bert_embedding.embed_size

        if num_layers > 1:
            self.lstm = LSTM(self.input_size, num_layers=num_layers, hidden_size=hidden_size,
                             bidirectional=True,
                             batch_first=True, dropout=dropout)
        else:
            self.lstm = LSTM(self.input_size, num_layers=num_layers, hidden_size=hidden_size,
                             bidirectional=True,
                             batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.out_fc = nn.Linear(hidden_size * 2, num_classes)
        self.crf = ConditionalRandomField(num_classes, include_start_end_trans=True)
        self.init_wight()

    @torch.no_grad()
    def init_wight(self):
        for n, p in self.named_parameters():
            if 'bert' not in n and 'embedding' not in n and 'pos' not in n and 'pe' not in n \
                    and 'bias' not in n and 'crf' not in n and p.dim() > 1:
                try:
                    if self.init == 'uniform':
                        nn.init.xavier_uniform_(p)
                    elif self.init == 'norm':
                        nn.init.xavier_normal_(p)
                except:
                    exit(1208)

    def forward(self, chars, bigrams, seq_len, target):
        mask = chars.ne(0)
        # batch_size = chars.size(0)
        # max_seq_len = chars.size(1)

        embedding = self.char_embed(chars)
        if self.use_bigram:
            bigrams_embedding = self.bigram_embed(bigrams)
            embedding = torch.cat([embedding, bigrams_embedding], dim=-1)
        if self.use_bert:
            bert_embeddings= self.bert_embedding(chars)
            embedding = torch.cat([embedding, bert_embeddings], dim=-1)

        embedding = self.dropout(embedding)
        feats,_ = self.lstm(embedding, seq_len=seq_len)
        feats = self.dropout(feats)
        feats = self.out_fc(feats)
        logits = F.log_softmax(feats, dim=-1)

        if self.training:
            loss = self.crf(logits, target, mask)
            return {'loss': loss}
        else:
            paths, score = self.crf.viterbi_decode(logits, mask) # 最优路径及对应的分数
            return {'pred': paths,'score':score}





