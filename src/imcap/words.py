from collections import namedtuple
from imcap.files import is_newer_than,read_lines
from imcap.stage import measure
from imcap import utils, files
import string, json
import numpy as np
from typing import *
DescMap = Dict[str, List[str]]
SeqInfo = namedtuple('SeqInfo', ['max_desc_size', 'vocab_size', 'tokenizer'])
ENDSEQ = 'endseq'
STARTSEQ = 'startseq'
def make_descmap(lines : List[str],separator : str, **kwargs) -> DescMap:
    '''
    min_occurence - if word has less than 5 occurences in text discard it and all sentences containing it
    max_len - if sentence is longer then max_len words, discard it 
    '''
    descs = dict()
    for line in lines:
        label, words = line.split(separator)
        words = words.split()
        if len(words) < 2: continue
        label = label.split('.')[0]
        descs.setdefault(label,[]).append(STARTSEQ+' ' + ' '.join(clean_wordlist(words)) +' '+ENDSEQ)
    return clean_descmap(descs,kwargs.get('min_occurence',5),kwargs.get('max_len',30))


def make_vocab(descmap: DescMap) -> Set[str]:
    all_words = []
    for desclist in descmap.values():
        desc = ' '.join(desclist)
        words = desc.split()
        all_words += words
    return set(all_words)

@measure("Sequencing descriptions")
def make_seqs(descmap: DescMap) -> Tuple[Dict[str,List[Tuple[List[List[int]],List[int]]]],SeqInfo]:
    from tensorflow.keras.preprocessing.text import Tokenizer

    all_desc_list = utils.flatten(descmap.values())
    max_desc_size = max(len(d.split()) for d in all_desc_list)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_desc_list)
    vocab_size = len(tokenizer.word_index)+1

    return {key:[utils.make_divs(s) for s in tokenizer.texts_to_sequences_generator(descs)] for key,descs in descmap.items()}, SeqInfo(max_desc_size, vocab_size, tokenizer)

def save_seqs(word_seqs, filepath:str, seqinfo: SeqInfo):
    obj = dict()
    obj['seqs'] = word_seqs
    obj['vsize'] = seqinfo.vocab_size if seqinfo.vocab_size is not None else 0
    obj['maxseq'] = seqinfo.max_desc_size if seqinfo.max_desc_size is not None else 0
    with open(filepath, "w") as write:
        json.dump(obj, write)

@measure("Loading sequences")
def load_seqs(infile, subset: Set[str] = None):
    print(f'Loading sequences from {infile}' + (f' subset size: {len(subset)}' if subset != None else ''))
    seqs = dict()
    with open(infile, "r") as read:
        seqs = json.load(read)
    if subset != None:
        seqs['seqs'] = {k:v for k,v in seqs['seqs'].items() if k in subset}
    return seqs['seqs'], seqs['maxseq'], seqs['vsize']


def clean_wordlist(wordlist: List[str]) -> List[str]:
    transtable = str.maketrans('','',string.punctuation) 
    c = [word.lower().translate(transtable) for word in wordlist]
    c = [word for word in c if len(word) > 1 and word.isalpha()]
    return c

def save_descmap(descmap: DescMap, filepath:str) -> None:
    with files.file(filepath) as write:
        json.dump( descmap , write )

@measure("Preprocessing captions")
def preprocess(infile
, descfile='words.json'
, seqfile='seqs.json'
, tokenfile='tokenizer.json'
, min_occur=5
, max_len=30):
    if files.age(infile) < files.age(descfile):
        print(f'Updating description file ({descfile})...')
        lines = read_lines(infile)
        desc = make_descmap(lines,'\t',min_occurence = min_occur, max_len=max_len)
        save_descmap(desc,descfile)
    else:
        desc = load_descmap(descfile)
        print(f'Descriptions are up to date ({descfile})...')
        
    if files.age(descfile) < files.age(seqfile):
        print(f'Updating sequences file ({seqfile})...')
        sqs,seqinfo = make_seqs(desc)
        save_seqs(sqs,seqfile,seqinfo)
        files.write(tokenfile,seqinfo.tokenizer.to_json())
    else:
        print(f'Sequences are up to date ({seqfile})...')
    

@measure("Loading descriptions")
def load_descmap(infile: str, subset : Set[str] = None) -> DescMap :
    dm : DescMap
    with open(infile, "r") as read:
        dm = json.load(read)
    if subset != None:
        dm = {k:v for k,v in dm.items() if k in subset}
    return dm

def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
def load_tokenizer(config_path: str):
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    return tokenizer_from_json(files.read(config_path))

def print_histogram(descmap: DescMap):
    histogram = utils.Counter()
    sent_lens = utils.Counter()
    for sentence in utils.flatten(descmap.values()):
        sent_lens[len(sentence.split())] +=1
        for word in sentence.split():
            histogram[word] += 1
    num_words = len(histogram)
    print(f'Num Words: {num_words}')
    histogram = {w:c for w,c in histogram.items()}
    for p in range(0,10,1):
        print(f'p={p}: {len([c for c in histogram.values() if c<p])}')
    for l,n in sorted(sent_lens.items()):
        print(f'len={l}: {n}')
    print(sum([c for l,c in sent_lens.items() if l > 21])) 

def clean_descmap(descmap: DescMap, min_occurence=0, max_len=21):
    #21 discards 5% of sentences while decreasing max sentence len from 72 (for f30k)
    histogram = utils.Counter()
    for sentence in utils.flatten(descmap.values()):
        for word in sentence.split():
            histogram[word] += 1
    banned = {w for w,c in histogram.items() if c < min_occurence}
    print(f'Banning {len(banned)} words from vocabulary')
    for label in descmap.keys():
        descmap[label] = [sent for sent in descmap[label] if len(banned.intersection(sent.split())) == 0 and len(sent.split()) <= max_len]

    descmap = {l:caps for l,caps in descmap.items() if len(caps) > 0}
    return descmap

    


    

