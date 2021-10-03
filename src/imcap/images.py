import zipfile,os,json
from imcap import stage,models,files
from typing import *
FeatMap = Dict[str,List[float]]
def preprocess(zippath, feat_extractor, outpath):
    if files.is_newer_than(zippath,outpath):
        print(f'Image features are up to date ({outpath})...')
        return
    print(f'Opening: {zippath}')
    z = zipfile.ZipFile(zippath)
    m = models.get_image_feature_extractor(feat_extractor)
    feats = dict()
    print(f'Beginning image processing')
    namelist = [n for n in z.namelist() if not is_dir(n)]
    bar = stage.ProgressBar("Processing files",len(namelist))
    for i in range(len(namelist)):
        path = z.extract(namelist[i])
        im_name = files.get_filename(path)
        bar.update(im_name)
        feat = preprocess_image(path,m,models.preproc[feat_extractor],models.expected_size[feat_extractor])
        feats[im_name] = feat.tolist()[0]
        os.remove(path)
    with files.write(outpath) as write:
        json.dump( feats , write )

@stage.measure("Loading features")
def load_featmap(infile:str, subset: Set[str]= None) -> FeatMap:
    print(f'Loading features from {infile}')
    fm : FeatMap
    with open(infile, "r") as read:
        fm = json.load(read)
    if subset != None:
        fm = {k:v for k,v in fm.items() if k in subset}
    return fm

def preprocess_image(filename: str, model_name: str):
    from tensorflow.keras.preprocessing.image import load_img
    from tensorflow.keras.preprocessing.image import img_to_array
    model = models.get_image_feature_extractor(model_name)
    prefunc = models.preproc[model_name]
    dst_size = models.expected_size[model_name]
    img = img_to_array(load_img(filename,target_size=dst_size))
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = prefunc(img)
    return model.predict(img, verbose=0)

def is_dir(name : str):
    return (name.endswith('/')) or (name.startswith('__MACOSX'))