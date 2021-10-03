import datetime
from typing import Callable
import numpy as np
from numpy.lib import utils
from imcap import stage, utils, words
from tensorflow.keras.applications import vgg16, vgg19
feat_extractors = ['VGG16', 'VGG19']
expected_size = {
    'VGG16': (224,224),
    'VGG19': (244,244)
}
preproc = {
    'VGG16': vgg16.preprocess_input,
    'VGG19': vgg19.preprocess_input
}
output_size = {
    'VGG16': 4096,
    'VGG19': 4096
}

@stage.measure("Loading feature extractor")
def get_image_feature_extractor(name: str):
    from tensorflow.keras.models import Model
    name = name.upper()
    if name == 'VGG16' or name == 'VGG-16':
        model = vgg16.VGG16()
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        return model
    elif name == 'VGG19' or name == 'VGG-19':
        model = vgg19.VGG19()
        model = Model(inputs = model.inputs, outputs=model.layers[-2].output)
        return model
    else:
        return None

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
    Xs1,Xs2,Ys = [],[],[]
    while True:
        for key, imfeat in image_feats.items():
            xs, ys = utils.unzip_xy_pairs(word_seqs[key])
            # xs = pad_sequences(xs, maxlen=seq_size)
            # ys = to_categorical(ys,num_classes=vocab_size)
            Xs1 = [imfeat]*len(xs)
            Xs2 = xs
            Ys = ys
            yield ([np.array(Xs1), pad_sequences(Xs2, maxlen=seq_size)], to_categorical(Ys, num_classes=vocab_size))


def seqs_to_vec(word_seqs, seq_size, vocab_size):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    allXs = []
    allYs = []
    for xy_pairs in word_seqs.values():
        Xs, Ys = utils.unzip_xy_pairs(xy_pairs)
        allXs += Xs
        allYs += Ys 
    return pad_sequences(allXs, maxlen=seq_size), to_categorical(allYs,num_classes=vocab_size)

@stage.measure("Constructing ANN model")
def make_model(seq_len,word_vec_len, feat_len, embed_vec_len=256):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input,Dense,LSTM,Embedding,Dropout,Add

    inputs1 = Input(shape=(feat_len,), name='fe_input')
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(embed_vec_len, activation='relu')(fe1)

    inputs2 = Input(shape=(seq_len,), name='seq_input')
    se1 = Embedding(input_dim=word_vec_len,
                    output_dim=embed_vec_len,
                    input_length=seq_len,
                    mask_zero=True,
                    name='embed_input')(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(embed_vec_len, unroll=True)(se2)

    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(embed_vec_len, activation='relu')(decoder1)
    outputs = Dense(word_vec_len, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'],steps_per_execution=2)
    return model

def get_callbacks(model_name='my_model',checkpt_dir='checkpoints'):
    import tensorflow as tf
    log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath= checkpt_dir + f'/{model_name}' + "_{epoch}",
        save_best_only=True,
        monitor='val_loss',
        verbose=1
        )
    return [tensorboard_callback, checkpoint_callback]

def save_model(model, dirname):
    import tensorflow as tf
    tf.keras.models.save_model(model,filepath=dirname)

def load_model(dirname):
    import tensorflow as tf
    return tf.keras.models.load_model(dirname)

def apply_desc_model(model, input, tokenizer, times: int) -> str:
    from tensorflow.python.keras.preprocessing.sequence import pad_sequences
    from numpy import argmax
    text = words.STARTSEQ
    for i in range(times):
        seq = tokenizer.texts_to_sequences([text])[0]
        seq = pad_sequences([seq],maxlen=times)
        y = model.predict([input,seq])
        y = argmax(y)
        word = words.word_for_id(y,tokenizer)
        if word is None or word == words.ENDSEQ:
            break
        text += ' ' + word
    return text
