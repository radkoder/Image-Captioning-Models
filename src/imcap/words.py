from imcap.files import is_newer_than,read_lines
from imcap.stage import measure
from imcap import utils
import string, json
import numpy as np

DescMap = dict[str, list[str]]
ENDSEQ = 'endseq'
STARTSEQ = 'startseq'
def make_descmap(lines : list[str],separator : str) -> DescMap:
    descs = dict()
    for line in lines:
        label, words = line.split(separator)
        words = words.split()
        if len(words) < 2: continue
        label = label.split('.')[0]
        descs.setdefault(label,[]).append(STARTSEQ+' ' + ' '.join(clean_wordlist(words)) +' '+ENDSEQ)
    return descs


def make_vocab(descmap: DescMap) -> set[str]:
    all_words = []
    for desclist in descmap.values():
        desc = ' '.join(desclist)
        words = desc.split()
        all_words += words
    return set(all_words)

@measure("Sequencing descriptions")
def make_seqs(descmap: DescMap) -> dict[str,list[tuple[list[list[int]],list[int]]]]:
    from tensorflow.keras.preprocessing.text import Tokenizer

    all_desc_list = utils.flatten(descmap.values())
    max_desc_size = max(len(d.split()) for d in all_desc_list)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_desc_list)
    vocab_size = len(tokenizer.word_index)+1

    return {key:[utils.make_divs(s) for s in tokenizer.texts_to_sequences_generator(descs)] for key,descs in descmap.items()}, max_desc_size, vocab_size

def save_seqs(word_seqs, filepath:str, v_size = None, max_desc = None):
    obj = dict()
    obj['seqs'] = word_seqs
    obj['vsize'] = v_size if v_size is not None else 0
    obj['maxseq'] = max_desc if max_desc is not None else 0
    with open(filepath, "w") as write:
        json.dump(obj, write)

@measure("Loading sequences")
def load_seqs(infile, subset: set[str] = None):
    seqs = dict()
    with open(infile, "r") as read:
        seqs = json.load(read)
    if subset != None:
        seqs['seqs'] = {k:v for k,v in seqs['seqs'].items() if k in subset}
    return seqs['seqs'], seqs['maxseq'], seqs['vsize']


def clean_wordlist(wordlist: list[str]) -> list[str]:
    transtable = str.maketrans('','',string.punctuation) 
    c = [word.lower().translate(transtable) for word in wordlist]
    c = [word for word in c if len(word) > 1 and word.isalpha()]
    return c

def save_descmap(descmap: DescMap, filepath:str) -> None:
    with open( filepath , "w" ) as write:
        json.dump( descmap , write )

@measure("Preprocessing captions")
def preprocess(infile, descfile='words.json', seqfile='seqs.json'):
    desc : DescMap
    if not is_newer_than(infile,descfile):
        print(f'Updating description file ({descfile})...')
        lines = read_lines(infile)
        desc = make_descmap(lines,'\t')
        save_descmap(desc,descfile)
    else:
        print(f'Descriptions are up to date ({descfile})...')
    if not is_newer_than(descfile, seqfile):
        print(f'Updating sequences file ({seqfile})...')
        sqs,slen,vsize = make_seqs(desc)
        save_seqs(sqs,seqfile,v_size=vsize, max_desc=slen)
    else:
        print(f'Sequences are up to date ({seqfile})...')
    

    

@measure("Loading descriptions")
def load_descmap(infile: str, subset : set[str] = None) -> DescMap :
    dm : DescMap
    with open(infile, "r") as read:
        dm = json.load(read)
    if subset != None:
        dm = {k:v for k,v in dm.items() if k in subset}
    return dm

