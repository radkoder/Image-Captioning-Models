import datetime
from typing import Callable
import numpy as np
from numpy.lib import utils

import tensorflow as tf
from imcap import stage, utils, words, files
from tensorflow.keras.applications import vgg16, vgg19


feat_extractors = ['VGG16', 'VGG19']
expected_size = {
    'VGG16': (224,224),
    'VGG19': (224,224)
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
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, mode='min', verbose=1)

    return [tensorboard_callback, earlystop_callback]

def save_model(model, dirname):
    import tensorflow as tf
    tf.keras.models.save_model(model,filepath=dirname)

def load_model(dirname) -> Model:
    import tensorflow as tf
    return tf.keras.models.load_model(dirname)

def release_model(model, dirname):
    import tensorflow as tf
    files.write(dirname+'/config.json', model.to_json())
    model.save_weights(dirname+'/weights.h5')

def load_release(dirname) -> Model:
    import tensorflow as tf
    model = tf.keras.models.model_from_json(files.read(dirname+'/config.json'))
    model.load_weights(dirname+'/weights.h5')
    return model

def apply_desc_model(model, input, tokenizer, log_endseq = False) -> str:
    from tensorflow.python.keras.preprocessing.sequence import pad_sequences
    from numpy import argmax
    text = words.STARTSEQ
    seq_len = model.get_layer('seq_input').input_shape[0][1]

    for i in range(seq_len):
        seq = tokenizer.texts_to_sequences([text])[0]
        seq = pad_sequences([seq],maxlen=seq_len)
        y = model.predict_on_batch([input,seq])
        y = argmax(y)
        word = words.word_for_id(y,tokenizer)
        if word is None or word == words.ENDSEQ:
            if log_endseq:
                print(f"Sequence ended in {word}")
            break
        text += ' ' + word

    return text[len(words.STARTSEQ)+1:]
