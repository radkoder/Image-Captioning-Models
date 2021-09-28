from imcap import main

data_dir = 'data'
feat_net = 'VGG16'
dataset_name = 'Flickr8k'
artifact_dir = f'artifacts/{dataset_name}-{feat_net}'

images_zip = f'{data_dir}/Flickr8k_Dataset.zip'
descriptions_file = f'{data_dir}/Flickr8k.token.txt'
trainset_file = f'{data_dir}/Flickr_8k.trainImages.txt'

seq_file = f'{artifact_dir}/seqs.json'
word_file = f'{artifact_dir}/words.json'
feat_file = f'{artifact_dir}/feats.json'


main.make_main(f'data {dataset_name}'.split())
main.make_main(f'feats {images_zip} {feat_net} {feat_file}'.split())
main.make_main(f'seqs {descriptions_file} {word_file} {seq_file}'.split())
main.train_main(f'--setfile={trainset_file} --featfile={feat_file} --seqfile={seq_file}'.split())
