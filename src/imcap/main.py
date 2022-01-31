import getopt,sys, itertools
from subprocess import call
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
    model_dir = f'models'
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
    train_epochs = 10
    desc_name = None

def make_config():
    if config.desc_name == None:
        print("No desc name in config")
        exit(-1)
    config.artifact_dir = f'artifacts/{config.dataset_name}-{config.feat_net}'
    config.fex_dir=f'{config.model_dir}/{config.feat_net}_feat_extractor'
    config.desc_dir=f'{config.model_dir}/{config.feat_net}_desc_net_{config.desc_name}'
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
        elif dataset_name == 'Flickr30k':
            retcode = call("python " + f"{config.data_dir}/f30k/make_sets.py", shell=True)
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

    # data,seqinfo = mlutils.load_set(setfile=kwargs.get('setfile'),
    #                         seqfile=kwargs.get('seqfile'),
    #                         featfile=kwargs.get('featfile'),
    #                         set_role="train")
    data,seqinfo = mlutils.load_set(setfile=kwargs.get('setfile'),
                            seqfile=kwargs.get('seqfile'),
                            featfile=kwargs.get('featfile'),
                            set_role="train",as_generator=True)
    valdata,_ = mlutils.load_set(setfile=kwargs.get('valset'),
                        seqfile=kwargs.get('seqfile'),
                        featfile=kwargs.get('featfile'),
                        set_role="eval", as_generator=True)
    model = models.make_model(seqinfo.max_desc_size,seqinfo.vocab_size,kwargs.get('featsize', 4096)) 
   
    #utils.print_sizes(("validation data",valdata), 
    #                 ("training data",data))

    tenq = tf.keras.utils.OrderedEnqueuer(data)
    tenq.start(max_queue_size=1000)
    train_gen = tenq.get()

    venq = tf.keras.utils.OrderedEnqueuer(valdata)
    venq.start(max_queue_size=1000)
    val_gen = venq.get()

    if kwargs.get('dry_run',False):
        print("Model fitting would happen here...")
    else:
        # model.fit(x=[data.X_feat,data.X_seq],y=data.Ys,
        # validation_data=([valdata.X_feat,valdata.X_seq],valdata.Ys),
        # epochs=kwargs.get('epochs',3),callbacks=models.get_callbacks())
        # models.save_model(model, kwargs.get('output_name','desc_net'))
        model.fit(x=train_gen,
        validation_data=val_gen,
        epochs=kwargs.get('epochs',1),
        steps_per_epoch=len(data), #see mlutils make_input_set_generator
        validation_steps=len(valdata),
        use_multiprocessing=False,
        workers=1,
        max_queue_size=10,
        callbacks=models.get_callbacks())
        models.save_model(model, kwargs.get('output_name','unnamed_desc_net'))
    
    venq.stop()
    tenq.stop()
    models.apply_desc_model(model,images.preprocess_image('rudzia.jpg',config.feat_net),mlutils.load_tokenizer(config.token_config))


def train():
    train_main(setfile=config.trainset_file,
               featfile=config.feat_file,
               seqfile=config.seq_file,
               output_name=config.desc_dir,
               featsize=models.output_size[config.feat_net],
               valset=config.evalset_file,
               epochs= config.train_epochs)

def apply(img_name :str, modelpath = config.desc_dir, featnet = None)-> str:
    if featnet == None: featnet = config.feat_net
    feats = images.preprocess_image(img_name,featnet)
    # valdata,seqinfo = mlutils.load_set(setfile=config.evalset_file,
    #                     seqfile=config.seq_file,
    #                     featfile=config.feat_file,
    #                     set_role="eval", as_generator=True, trim=1)
    
    # desc = models.make_model(seqinfo.max_desc_size,seqinfo.vocab_size,4096)
    desc = models.load_release(modelpath+'_release')
    token = mlutils.load_tokenizer(config.token_config)
    #desc.reset_states()
    return models.apply_desc_model(desc,feats,token)

@stage.measure("Captioning images")
def apply_batch(img_name , modelpath = config.desc_dir, featnet = None):
    if featnet == None: featnet = config.feat_net
    feats = images.preprocess_images(img_name,featnet)
    # valdata,seqinfo = mlutils.load_set(setfile=config.evalset_file,
    #                     seqfile=config.seq_file,
    #                     featfile=config.feat_file,
    #                     set_role="eval", as_generator=True, trim=1)
    
    # desc = models.make_model(seqinfo.max_desc_size,seqinfo.vocab_size,4096)
    desc = models.load_release(modelpath+'_release')
    token = mlutils.load_tokenizer(config.token_config)
    strings = dict()
    for f,n in zip(feats,img_name):
        strings[n]= models.apply_desc_model(desc,f,token)
    return strings

@stage.measure("Calculating BLEU for model")
def test(modelpath = config.desc_dir):
    from nltk.translate.bleu_score import corpus_bleu
    model = models.load_release(modelpath+'_release')
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
        generated = models.apply_desc_model(model,np.array([featmap[label]]),tokenizer)
        refs = [d.split() for d in desclist]
        references.append(refs)
        hypotesis.append(generated.split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(references, hypotesis, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(references, hypotesis, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(references, hypotesis, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(references, hypotesis, weights=(0.25, 0.25, 0.25, 0.25)))

def release():
    m = models.load_model(config.desc_dir)
    models.release_model(m,config.desc_dir+'_release')
    
def main(argv=sys.argv):
    return      

# if __name__ == '__main__':
#     main()
