from fastNLP import Vocabulary
import os
from fastNLP import cache_results
from fastNLP.embeddings import StaticEmbedding
# from fastNLP_module import StaticEmbedding
from fastNLP.io.loader import ConllLoader

def get_bigrams(words):
    result = []
    for i,w in enumerate(words):
        if i!=len(words)-1:
            result.append(words[i]+words[i+1])
        else:
            result.append(words[i]+'<end>')

    return result

@cache_results(_cache_fp='cache/tianchiner',_refresh=False)
def load_tianchiner(path,char_embedding_path=None,bigram_embedding_path=None,index_token=True,
                       char_min_freq=1,bigram_min_freq=1,only_train_min_freq=0):

    train_path = os.path.join(path,'train.conll')
    dev_path = os.path.join(path,'dev.conll')
    test_path = os.path.join(path,'test.conll')
    #test_path1 = os.path.join(path, 'test1.conll')

    loader = ConllLoader(['chars','target'])
    train_bundle = loader.load(train_path)
    dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)
    #test_bundle1 = loader.load(test_path1)


    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['dev'] = dev_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']
    #datasets['test1'] = test_bundle1.datasets['train']


    datasets['train'].apply_field(get_bigrams,field_name='chars',new_field_name='bigrams')
    datasets['dev'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    #datasets['test1'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    datasets['dev'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')
    #datasets['test1'].add_seq_len('chars')

    for k, v in datasets.items():
        v.set_input('chars', 'bigrams')
        v.set_target('target','seq_len')

    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()
    print(datasets.keys())
    print(len(datasets['dev']))
    print(len(datasets['test']))
    print(len(datasets['train']))
    char_vocab.from_dataset(datasets['train'],field_name='chars',
                            no_create_entry_dataset=[datasets['dev'],datasets['test']])
    bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'],datasets['test']])
    label_vocab.from_dataset(datasets['train'],field_name='target')
    if index_token:
        char_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='chars',new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='target',new_field_name='target')

    vocabs = {}
    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab
    vocabs['bigram'] = bigram_vocab


    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab,char_embedding_path,word_dropout=0.01,
                                         min_freq=char_min_freq,only_train_min_freq=only_train_min_freq)
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab,bigram_embedding_path,word_dropout=0.01,
                                           min_freq=bigram_min_freq,only_train_min_freq=only_train_min_freq)
        embeddings['bigram'] = bigram_embedding

    vocabs['char'].index_dataset(* (datasets.values()),
                             field_name='chars', new_field_name='chars')
    vocabs['bigram'].index_dataset(* (datasets.values()),
                               field_name='bigrams', new_field_name='bigrams')
    vocabs['label'].index_dataset(* (datasets.values()),
                              field_name='target', new_field_name='target')



    return datasets,vocabs,embeddings

if __name__ == '__main__':
    pass