import datetime
import numpy as np
from numpy.lib import utils
from imcap import stage, utils 

feat_extractors = ['VGG16']
expected_size = {
    'VGG16': (224,224)
}
@stage.measure("Loading feature extractor")
def get_image_feature_extractor(name):
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications.vgg16 import VGG16
    if name == 'VGG16' or name == 'VGG-16':
        model = VGG16()
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
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
def make_model(seq_len,word_vec_len, feat_len):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input,Dense,LSTM,Embedding,Dropout,Add
    
    inputs1 = Input(shape=(feat_len,), name='fe_input')
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(seq_len,), name='seq_input')
    se1 = Embedding(word_vec_len, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(word_vec_len, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_callbacks():
    import tensorflow as tf
    log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)
    return [tensorboard_callback]
