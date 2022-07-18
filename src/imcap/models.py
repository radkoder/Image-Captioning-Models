import datetime
from typing import Callable
import numpy as np
from numpy.lib import utils
from tensorflow.keras import Model
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
VGG16_model = None
VGG19_model = None
@stage.measure("Loading feature extractor")
def get_image_feature_extractor(name: str):
    from tensorflow.keras.models import Model
    name = name.upper()
    if name == 'VGG16' or name == 'VGG-16':
        if VGG16_model != None:
            return VGG16_model
        model = vgg16.VGG16()
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        return model
    elif name == 'VGG19' or name == 'VGG-19':
        if VGG19_model != None:
            return VGG19_model
        model = vgg19.VGG19()
        model = Model(inputs = model.inputs, outputs=model.layers[-2].output)
        return model
    else:
        return None


@stage.measure("Constructing ANN model")
def make_model(seq_len,vocab_size, feat_len, embed_vec_len=256):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input,Dense,LSTM,Embedding,Dropout,Add

    inputs1 = Input(shape=(feat_len,), name='fe_input')
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(embed_vec_len, activation='relu')(fe1)

    inputs2 = Input(shape=(seq_len,), name='seq_input')
    se1 = Embedding(input_dim=vocab_size,
                    output_dim=embed_vec_len,
                    input_length=seq_len,
                    mask_zero=True,
                    name='embed_input')(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(embed_vec_len, unroll=True)(se2)

    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(embed_vec_len, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    #tweak epsilon value in Adam optimizer to higher value 0.1 or 1.0
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_callbacks(model_name='my_model',checkpt_dir='checkpoints'):
    import tensorflow as tf
    log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath= f'{checkpt_dir}/{model_name}' + "_{epoch}",
        save_best_only=True,
        monitor='val_loss',
        verbose=1
        )
    return [tensorboard_callback, checkpoint_callback]

def save_model(model, dirname):
    import tensorflow as tf
    tf.keras.models.save_model(model,filepath=dirname)

def load_model(dirname) -> Model:
    import tensorflow as tf
    return tf.keras.models.load_model(dirname)

def apply_desc_model(model, input, tokenizer, times: int, log_endseq = False) -> str:
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
            if log_endseq:
                print(f"Sequence ended in {word}")
            break
        text += ' ' + word

    return text[len(words.STARTSEQ)+1:]
