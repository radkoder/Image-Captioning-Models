import zipfile,os,json
import models,files
from stage import *
FeatMap = dict[str,list[float]]
def preprocess(zippath, feat_extractor, outpath):
    from tensorflow.keras.preprocessing.image import load_img
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.applications.vgg16 import preprocess_input

    print(f'Opening: {zippath}')
    z = zipfile.ZipFile(zippath)
    m = models.get_image_feature_extractor()
    feats = dict()
    print(f'Beginning image processing')
    namelist = z.namelist()
    bar = ProgressBar("Processing files",len(namelist))
    for i in range(len(namelist)):
        if namelist[i].endswith('/'): continue
        path = z.extract(namelist[i])
        im_name = files.get_filename(path)
        bar.update(im_name)
        img = img_to_array(load_img(path,target_size=models.expected_size[feat_extractor]))
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = preprocess_input(img)
        feat = m.predict(img, verbose=0)
        
        feats[im_name] = feat.tolist()[0]
        os.remove(path)
    with open( outpath , "w" ) as write:
        json.dump( feats , write )

@measure("Loading features")
def load_featmap(infile:str, subset: set[str]= None) -> FeatMap:
    fm : FeatMap
    with open(infile, "r") as read:
        fm = json.load(read)
    if subset != None:
        fm = {k:v for k,v in fm.items() if k in subset}
    return fm