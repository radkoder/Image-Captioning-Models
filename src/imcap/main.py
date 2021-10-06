import getopt,sys, itertools

from tensorflow.python.keras.backend import update
from imcap.files import unpack
from imcap import *
import numpy as np
class config():
    data_dir = 'data'
    feat_net = None
    dataset_name = None
    test_image = None
    artifact_dir = f'artifacts/{dataset_name}-{feat_net}'
    model_dir = f'models/'
    images_zip = None
    descriptions_file = None
    trainset_file = None
    testset_file = None
    evalset_file = None
    fex_dir=f'{model_dir}/{feat_net}_feat_extractor'
    desc_dir=f'{model_dir}/{feat_net}_desc_net'
    seq_file = f'{artifact_dir}/seqs.json'
    word_file = f'{artifact_dir}/words.json'
    feat_file = f'{artifact_dir}/feats.json'
    token_config = f'{artifact_dir}/tokenizer.json'

def make_config():
    config.artifact_dir = f'artifacts/{config.dataset_name}-{config.feat_net}'
    config.fex_dir=f'{config.model_dir}/{config.feat_net}_feat_extractor'
    config.desc_dir=f'{config.model_dir}/{config.feat_net}_desc_net_overfit_test'
    config.seq_file = f'{config.artifact_dir}/seqs.json'
    config.word_file = f'{config.artifact_dir}/words.json'
    config.feat_file = f'{config.artifact_dir}/feats.json'
    config.token_config = f'{config.artifact_dir}/tokenizer.json'

def make_main(arg : str):
    #make fex-model
    if(arg == 'fex-model'):
        print(f"Creating feature extraction model ({config.feat_net})")
        model = models.get_image_feature_extractor(config.feat_net)
        path = config.fex_dir
        model.save(path)
        print(f'Model saved in {path}')
    #make feats 
    elif(arg == 'feats'):
        print(f"Creating features from images {config.images_zip} -> ({config.feat_net}) -> {config.feat_file} ")
        images.preprocess(config.images_zip, config.feat_net, config.feat_file)
    #make seqs descriptions.txt-> ->descfile.json ->seqfile.json
    elif(arg == 'seqs'):
        print(f"Creating token sequences  {config.descriptions_file} -> [descriptions={config.word_file}, sequences={config.seq_file}, tokenizer={config.token_config}] ")
        words.preprocess(config.descriptions_file, config.word_file, config.seq_file, config.token_config)
    #make data Flickr8k
    elif(arg == 'data'):
        dataset_name = config.dataset_name
        if dataset_name == 'Flickr8k':
            unpack(f'data/{dataset_name}_text.zip',['Flickr_8k.testImages.txt','Flickr8k.token.txt', 'Flickr_8k.trainImages.txt', 'Flickr_8k.devImages.txt'])
        else:
            print(f"Unknown dataset {dataset_name}")
    else:
        print(f"primary instruction unknown: {arg}")
        
@stage.measure("Making artifacts")
def make(args: str):
    for a in args.split():
        make_main(a)

def train_main(**kwargs):
    '''
    setfile - file describing data subset 

    featfile - file describing images features 

    seqfile- file describing text sequences

    output_nem - dir to save tensorflow model 

    featsize - size of the image feature vector

    dry_run - do not fit the model
    '''
    import tensorflow as tf
    # proceed = input("This may take a lot of time are you sure you want to proceed [y/N]: ")
    # if proceed.lower() == 'n':
    #     print("Aborting.")
    #     exit(0)

    data,seqinfo = mlutils.load_set(setfile=kwargs.get('setfile'),
                            seqfile=kwargs.get('seqfile'),
                            featfile=kwargs.get('featfile'),
                            set_role="train")
    valdata,_ = mlutils.load_set(setfile=kwargs.get('valset'),
                        seqfile=kwargs.get('seqfile'),
                        featfile=kwargs.get('featfile'),
                        set_role="eval")
    model = models.make_model(seqinfo.max_desc_size,seqinfo.vocab_size,kwargs.get('featsize', 4096)) 
   
    utils.print_sizes(("validation data",valdata), 
                      ("training data",data))
    if kwargs.get('dry_run',False):
        print("Model fitting would happen here...")
    else:
        model.fit(x=[data.X_feat,data.X_seq],y=data.Ys,
        validation_data=([valdata.X_feat,valdata.X_seq],valdata.Ys),
        epochs=10,callbacks=models.get_callbacks())
        models.save_model(model, kwargs.get('output_name','desc_net'))

def train():
    train_main(setfile=config.trainset_file,
               featfile=config.feat_file,
               seqfile=config.seq_file,
               output_name=config.desc_dir,
               featsize=models.output_size[config.feat_net],
               valset=config.evalset_file)

def apply(img_name :str)-> str:
    feats = images.preprocess_image(img_name,config.feat_net)
    desc = models.load_model(config.desc_dir)
    token = mlutils.load_tokenizer(config.token_config)
    return models.apply_desc_model(desc,feats,token,34)
@stage.measure("Calculating BLEU for model")
def test():
    from nltk.translate.bleu_score import corpus_bleu
    model = models.load_model(config.desc_dir)
    test_set = files.load_setfile(config.testset_file)
    descmap = words.load_descmap(config.word_file,test_set)
    featmap = images.load_featmap(config.feat_file,test_set)
    tokenizer = mlutils.load_tokenizer(config.token_config)
    references = []
    hypotesis = []
    bar = stage.ProgressBar("Calculating BLEU score", len(test_set))
    for label,desclist in descmap.items():
        bar.update(label)
        model.reset_states()
        generated = models.apply_desc_model(model,np.array([featmap[label]]),tokenizer,34)
        refs = [d.split() for d in desclist]
        references.append(refs)
        hypotesis.append(generated.split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(references, hypotesis, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(references, hypotesis, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(references, hypotesis, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(references, hypotesis, weights=(0.25, 0.25, 0.25, 0.25)))


def main(argv=sys.argv):
    return      
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


    model = models.make_model(seq_size,vocab_size,4096)

    # filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.pb'
    # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # gen = models.make_input_set_generator(word_seqs,image_set,vocab_size,seq_size)
    # model.fit(x=gen,y=None,steps_per_epoch=len(train_set),epochs=5,workers=4,callbacks=[checkpoint])

    X1,X2,Y = models.make_input_set(word_seqs,image_set,vocab_size,seq_size)
    model.fit(x=[X1,X2],y=Y,epochs=6,workers=4)
    model.save('word_generation_model_final')
    # Xs, Ys = next(gen)
    # print(f'Generated {len(Ys)} samples')
    # print(f'Sample dimentions: image features: {Xs[0].shape} \t word vectors: {Xs[1].shape} \t result vector: {Ys.shape}')

    # imXs, wdXs, Ys = models.make_input_set(word_seqs,image_set,vocab_size,seq_size)
    # print(f'Generated {len(Ys)} samples')
    # print(f'Sample dimentions: image features: {imXs.shape} \t word vectors: {wdXs.shape} \t result vector: {Ys.shape}')

# if __name__ == '__main__':
#     main()