from imcap import main

data_dir = 'data'
feat_net = 'VGG16'
dataset_name = 'Flickr8k'
test_image = 'D:/download/rudzia.jpg'
artifact_dir = f'artifacts/{dataset_name}-{feat_net}'
model_dir = f'models/'
images_zip = f'{data_dir}/Flickr8k_Dataset.zip'
descriptions_file = f'{data_dir}/Flickr8k.token.txt'
trainset_file = f'{data_dir}/Flickr_8k.testImages.txt'


fex_dir='D:\misc\wat\inz\models\word_generation_model_final'#f'{model_dir}/{feat_net}_feat_extractor'
desc_dir=f'{model_dir}/{feat_net}_desc_net'
seq_file = f'{artifact_dir}/seqs.json'
word_file = f'{artifact_dir}/words.json'
feat_file = f'{artifact_dir}/feats.json'
token_config = f'{artifact_dir}/tokenizer.json'


main.make_main(f'data {dataset_name}'.split())
main.make_main(f'feats {images_zip} {feat_net} {feat_file}'.split())
main.make_main(f'seqs {descriptions_file} {word_file} {seq_file} {token_config}'.split())
main.train_main(f'--setfile={trainset_file} --featfile={feat_file} --seqfile={seq_file} --output={desc_dir}'.split())
main.apply_main(f'{test_image} --fex-name={feat_net} --model={desc_dir} --tokenizer={token_config}'.split())
