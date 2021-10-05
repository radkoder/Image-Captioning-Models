from imcap import main
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
#tf.config.run_functions_eagerly()


main.config.feat_net = 'VGG16'
main.config.dataset_name = 'Flickr8k'
main.config.images_zip = f'{main.config.data_dir}/Flickr8k_Dataset.zip'
main.config.descriptions_file = f'{main.config.data_dir}/Flickr8k.token.txt'
main.config.trainset_file = f'{main.config.data_dir}/Flickr_8k.trainImages.txt'
main.config.evalset_file = f'{main.config.data_dir}/Flickr_8k.devImages.txt'
main.config.testset_file = f'{main.config.data_dir}/Flickr_8k.testImages.txt'
main.make_config()
main.make(f'data feats seqs')
main.train()
test_image = 'D:/download/rudzia.jpg'
result = main.apply(test_image)
print(result)
main.test()
