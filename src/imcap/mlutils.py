'''
Utility functons aggregating functinoality from other modules
'''
from collections import namedtuple
from imcap import words,utils,stage,files,images,models
from typing import Tuple
import itertools,math
import numpy as np
from imcap.words import SeqInfo

import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

DataTuple = namedtuple("DataTuple",["X_feat", "X_seq", "Ys"])

class TrainSequence(tensorflow.keras.utils.Sequence):
    def __init__(self,seqs,feats, vocab_size, seq_size, batch_size=128):
        self.Xs1, self.Xs2, self.Ys = [],[],[]
        self.vocab_size, self.seq_size, self.batch_size = vocab_size, seq_size,batch_size
        for key, seq in seqs.items():
                xs, ys = utils.unzip_xy_pairs(seq)
                # xs = pad_sequences(xs, maxlen=seq_size)
                # ys = to_categorical(ys,num_classes=vocab_size)
                self.Xs1 += [feats[key]]*len(xs)
                self.Xs2 += xs
                self.Ys += ys
    def setBatchSize(self,size):
        self.batch_size = size
            
    def __len__(self):
        return math.ceil(len(self.Ys)/self.batch_size)

    def __getitem__(self,index):
        batch_x1 = self.Xs1[index * self.batch_size : (index+1)*self.batch_size]
        batch_x2 = self.Xs2[index * self.batch_size : (index+1)*self.batch_size]
        batch_y = self.Ys[index * self.batch_size : (index+1)*self.batch_size]
        return ([np.array(batch_x1), pad_sequences(batch_x2, maxlen=self.seq_size)], to_categorical(batch_y, num_classes=self.vocab_size))
        


@stage.measure("Consolidating dataset")
def make_input_set(word_seqs, image_feats, vocab_size,seq_size):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    Xs1,Xs2,Ys = [],[],[]
    for key, imfeat in image_feats.items():
        xs, ys = utils.unzip_xy_pairs(word_seqs[key])
        # xs = pad_sequences(xs, maxlen=seq_size)
        # ys = to_categorical(ys,num_classes=vocab_size)
        Xs1 += [imfeat]*len(xs)
        Xs2 += xs
        Ys += ys
    return np.array(Xs1), pad_sequences(Xs2, maxlen=seq_size), to_categorical(Ys, num_classes=vocab_size)

#@utils.threadsafe_generator
def make_input_set_generator(word_seqs, image_feats, vocab_size,seq_size,group_size=1):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    import gc
    
    while True:
        for key_imfeat_pairs in utils.grouper(image_feats.items(),group_size,None):
            if key_imfeat_pairs == None:
                break
            Xs1,Xs2,Ys = [],[],[]
            for key, imfeat in key_imfeat_pairs:
                xs, ys = utils.unzip_xy_pairs(word_seqs[key])
                # xs = pad_sequences(xs, maxlen=seq_size)
                # ys = to_categorical(ys,num_classes=vocab_size)
                Xs1 += [imfeat]*len(xs)
                Xs2 += xs
                Ys += ys
            yield ([np.array(Xs1), pad_sequences(Xs2, maxlen=seq_size)], to_categorical(Ys, num_classes=vocab_size))

def load_set(setfile, seqfile, featfile, set_role : str= '',trim = None, as_generator = False) -> Tuple[DataTuple, words.SeqInfo]:
    '''
    trim - pass an int to trim the data set to given length
    '''
    train_set = files.load_setfile(setfile)
    if trim != None:
        train_set = set(itertools.islice(train_set, trim))
    word_seqs,sentence_len,vocab_size = words.load_seqs(seqfile,subset=train_set)
    image_set = images.load_featmap(featfile,subset=train_set)

    print(f'{set_role.capitalize()} set length: {len(image_set)}')
    print(f'Max seq size is: {sentence_len} words')
    print(f'Vocabulary size: {vocab_size} words')
    if as_generator:
        return TrainSequence(word_seqs,image_set,vocab_size,sentence_len), words.SeqInfo(sentence_len,vocab_size,None)
    else:
        return DataTuple(*make_input_set(word_seqs,image_set,vocab_size,sentence_len)), words.SeqInfo(sentence_len,vocab_size,None)

def load_tokenizer(tokenizer_config_file: str):
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    return tokenizer_from_json(files.read(tokenizer_config_file))

def set_len(setfile, trim=None):
    train_set = files.load_setfile(setfile)
    if trim != None:
        train_set = set(itertools.islice(train_set, trim))
    return len(train_set)