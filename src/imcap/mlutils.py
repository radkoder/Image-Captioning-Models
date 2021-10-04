'''
Utility functons aggregating functinoality from other modules
'''
from collections import namedtuple
from imcap import words,utils,stage,files,images,models
from typing import Tuple
import itertools
import numpy as np

from imcap.words import SeqInfo
DataTuple = namedtuple("DataTuple",["X_feat", "X_seq", "Ys"])

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

@utils.threadsafe_generator
def make_input_set_generator(word_seqs, image_feats, vocab_size,seq_size):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    
    while True:
        for key_imfeat_pairs in utils.grouper(image_feats.items(),100,None):
            Xs1,Xs2,Ys = [],[],[]
            for key, imfeat in key_imfeat_pairs:
                xs, ys = utils.unzip_xy_pairs(word_seqs[key])
                # xs = pad_sequences(xs, maxlen=seq_size)
                # ys = to_categorical(ys,num_classes=vocab_size)
                Xs1 += [imfeat]*len(xs)
                Xs2 += xs
                Ys += ys
            yield ([np.array(Xs1), pad_sequences(Xs2, maxlen=seq_size)], to_categorical(Ys, num_classes=vocab_size))
def load_set(setfile, seqfile, featfile, set_role : str= '',trim = None) -> Tuple[DataTuple, words.SeqInfo]:
    '''
    trim - pass an int to trim the data set to given lenth
    '''
    train_set = files.load_setfile(setfile)
    if trim != None:
        train_set = set(itertools.islice(train_set, trim))
    word_seqs,sentence_len,vocab_size = words.load_seqs(seqfile,subset=train_set)
    image_set = images.load_featmap(featfile,subset=train_set)

    print(f'{set_role.capitalize()} set length: {len(image_set)}')
    print(f'Max seq size is: {sentence_len} words')
    print(f'Vocabulary size: {vocab_size} words')
    return DataTuple(*make_input_set(word_seqs,image_set,vocab_size,sentence_len)), words.SeqInfo(sentence_len,vocab_size,None)
