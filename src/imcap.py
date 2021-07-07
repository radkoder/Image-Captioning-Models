import getopt,sys,zipfile,os,json
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import progressbar
def print_bar(max,i,what,status,length=20):
    r = int(i*length/max)
    s = '.'*r + ' '*(length-r)
    print(len(f'{what}:[{s}][{i}/{max}] => {status}          ')*' ',end='\r')
    print(f'{what}:[{s}][{i}/{max}] => {status}',end='\r')

feat_extractors = ['VGG16']
feat_used = 'VGG16'
expected_size = {
    'VGG16': (224,224)
}
def get_image_feature_extractor():
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    return model

def preprocess_images(zippath):
    print(f'Opening: {zippath}')
    z = zipfile.ZipFile(zippath)
    m = get_image_feature_extractor()
    feats = dict()
    print(f'Beginning image processing')
    namelist = z.namelist()
    for i in range(len(namelist)):
        if namelist[i].endswith('/'): continue
        path = z.extract(namelist[i])
        print_bar(len(namelist),i,"Processing files",path)
        img = img_to_array(load_img(path,target_size=expected_size[feat_used]))
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = preprocess_input(img)
        feat = m.predict(img, verbose=0)
        im_name = os.path.split(path)[1].split('.')[0]
        feats[im_name] = feat.tolist()[0]
        os.remove(path)
    with open( "VGG16-feats.json" , "w" ) as write:
        json.dump( feats , write )


    
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

preprocess_images('../data/Flickr8k_Dataset.zip')