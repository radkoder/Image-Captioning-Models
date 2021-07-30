import getopt,sys
import words, files, images, models
import utils as my_utils
from tensorflow.keras.callbacks import ModelCheckpoint
def build_model():
    pass
def train_main():
    pass
def apply_main():
    pass
def build_image_feature_model():
    pass

def build_word_encoding_model():
    pass

#Takes about an two hours
#images.preprocess('../data/Flickr8k_Dataset.zip')
# -> output in VGG16-feats.json

#Takes 5 s 
words.preprocess('../data/Flickr8k.token.txt',"words.json",'seqs.json')
# -> output in words.json

train_set = files.load_setfile('../data/Flickr_8k.trainImages.txt')
print(f'Loading dataset of {len(train_set)} examples')

word_seqs,seq_size,vocab_size = words.load_seqs('seqs.json',subset=train_set)
image_set = images.load_featmap('VGG16-feats.json',subset=train_set)
print(f'Image train set length: {len(image_set)}')
print(f'Max seq size is: {seq_size} words')
print(f'Vocabulary size: {vocab_size} words')

gen = models.make_input_set_generator(word_seqs,image_set,vocab_size,seq_size)
model = models.make_model(seq_size,vocab_size,4096)

filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.pb'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

model.fit(x=gen,y=None,steps_per_epoch=len(train_set),epochs=10,workers=4,callbacks=[checkpoint])

model.save('word_generation_model_fin')
# Xs, Ys = next(gen)
# print(f'Generated {len(Ys)} samples')
# print(f'Sample dimentions: image features: {Xs[0].shape} \t word vectors: {Xs[1].shape} \t result vector: {Ys.shape}')

# imXs, wdXs, Ys = models.make_input_set(word_seqs,image_set,vocab_size,seq_size)
# print(f'Generated {len(Ys)} samples')
# print(f'Sample dimentions: image features: {imXs.shape} \t word vectors: {wdXs.shape} \t result vector: {Ys.shape}')

ls