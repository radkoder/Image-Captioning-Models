import glob,os
from imcap import main
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
main.config.desc_name='f30k_finalattempt'
main.make_config()
files = glob.glob('examples/*.jpg')
result = main.apply_batch(files,
                        'models/VGG16_desc_net_f30k_finalattempt',
                        'VGG16')
for name,label in result.items():
    print(f'{name} : {label}')