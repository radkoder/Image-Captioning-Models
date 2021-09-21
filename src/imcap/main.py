import getopt,sys
from imcap.files import unpack
from imcap import *

@stage.measure("Making artifacts")
def make_main(args):
    #make fex-model [name]
    if(args[0] == 'fex-model'):
        print(f"Creating feature extraction model ({args[1]})")
        model = models.get_image_feature_extractor(args[1])
        path = f'../models/{args[1]}-feat-extractor'
        model.save(path)
        print(f'Model saved in {path}')
    #make vocab [filename]
    elif(args[0] == 'vocab'):
        print(f"Creating vocabulary index from {args[1]}")
        from tensorflow.keras.preprocessing.text import Tokenizer
        lines = files.read_lines('../data/Flickr8k.token.txt')
        desc = words.make_descmap(lines,'\t')
        all_desc_list = utils.flatten(desc.values())
        max_desc_size = max(len(d.split()) for d in all_desc_list)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_desc_list) 
        with open('../models/flickr8k.token.json', "w") as f:
            f.write(tokenizer.to_json())
            f.close()
    #make feats images.zip-> VGG-16 ->vgg16-feats.json
    elif(args[0] == 'feats'):
        print(f"Creating features from images {args[1]} -> ({args[2]}) -> {args[3]} ")
        images.preprocess(args[1], args[2], args[3])
    #make seqs descriptions.txt-> ->descfile.json ->seqfile.json
    elif(args[0] == 'seqs'):
        print(f"Creating token sequences  {args[1]} -> [{args[2]}, {args[3]}] ")
        words.preprocess(args[1], args[2], args[3])
    #make data Flickr8k
    elif(args[0] == 'data'):
        dataset_name = args[1]
        if dataset_name == 'Flickr8k':
            unpack('../data/{dataset_name}_text.zip',['Flickr_8k.testImages.txt','Flickr8k.token.txt'])
        


def train_main(args):
    '''
    --setfile=[file] describing data subset 
    --featfile=[file] describing images features 
    --seqfile=[file] describing text sequences
    --output=[dir] dir to save tensorflow model 
    --dry-run - do not fit the model


    '''
    proceed = input("This may take a lot of time are you sure you want to proceed [y/N]: ")
    if proceed.lower() == 'n':
        print("Aborting.")
        exit(0)
    opts, params = getopt.getopt(args, '', ['setfile=','featfile=','seqfile=','name=','output=','dry-run'])
    opts = dict(opts)
    setfile = opts.get('--setfile', 'train_set.txt')
    featfile = opts.get('--featfile','feats.json')
    seqfile = opts.get('--seqfile', 'seqs.json')
    out_name = opts.get('--output','desc_net')
    print(f'Reading setfile: {setfile}')
    train_set = files.load_setfile(setfile)
    print(f'Loading dataset of {len(train_set)} examples')

    print(f'Loading sequences from {seqfile}')
    word_seqs,seq_size,vocab_size = words.load_seqs(seqfile,subset=train_set)

    print(f'Loading features from {featfile}')
    image_set = images.load_featmap(featfile,subset=train_set)

    print(f'Image train set length: {len(image_set)}')
    print(f'Max seq size is: {seq_size} words')
    print(f'Vocabulary size: {vocab_size} words')

    model = models.make_model(seq_size,vocab_size,4096) #TODO parametrize

    X1,X2,Y = models.make_input_set(word_seqs,image_set,vocab_size,seq_size)
    if('--dry-run' in opts.keys()):
        print("Model fitting would happen here...")
    else:
        model.fit(x=[X1,X2],y=Y,epochs=6,workers=4,callbacks=models.get_callbacks())
        model.save(out_name)

    
def main(argv=sys.argv):
    if argv[1] == 'make':
        make_main(sys.argv[2:])
    if argv[1] == 'train':
        train_main(sys.argv[2:])
    else: 
        print("Unknown operation")
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