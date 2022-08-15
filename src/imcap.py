
from imcap import main,words
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf


main.config.feat_net = 'VGG16'

# main.config.dataset_name = 'Flickr8k'
# main.config.desc_name = 'f8k_archtests' #f8k release

# main.config.dataset_name = 'Flickr30k'
# main.config.desc_name = 'f30k_finalattempt' #f30k release

main.config.dataset_name = 'Flickr8k'
main.config.desc_name = 'f8k_gru_cm1024' 


if main.config.dataset_name == 'Flickr8k':   
    #Flickr8k
    main.config.images_zip = f'{main.config.data_dir}/Flickr8k_Dataset.zip'
    main.config.descriptions_file = f'{main.config.data_dir}/Flickr8k.token.txt'
    main.config.trainset_file = f'{main.config.data_dir}/Flickr_8k.trainImages.txt'
    main.config.evalset_file = f'{main.config.data_dir}/Flickr_8k.devImages.txt'
    main.config.testset_file = f'{main.config.data_dir}/Flickr_8k.testImages.txt'
elif main.config.dataset_name == 'Flickr30k':  
    #Flickr30k
    main.config.images_zip = f'{main.config.data_dir}/f30k/flickr30k-images.zip'
    main.config.descriptions_file = f'{main.config.data_dir}/f30k/descriptions.token'
    main.config.trainset_file = f'{main.config.data_dir}/f30k/f30k_trainImages.txt'
    main.config.evalset_file = f'{main.config.data_dir}/f30k/f30k_devImages.txt'
    main.config.testset_file = f'{main.config.data_dir}/f30k/f30k_testImages.txt'


main.make_config()
#main.make(f'data feats seqs')
#main.train()
#main.test()

def print_examples():
    import glob
    files = glob.glob('examples/*.jpg')
    result = main.apply_batch(files,main.config.desc_dir)
    for name,label in result.items():
        print(f'{name} : {label}')

def train_and_save():
    main.make('data feats seqs')
    main.train()
    main.release()
def BLEU():
    main.test(main.config.desc_dir)


#words.print_histogram(words.load_descmap(main.config.word_file))
train_and_save()
main.release()
print_examples()
BLEU()
#main.summary()

#main.make('data feats seqs')
