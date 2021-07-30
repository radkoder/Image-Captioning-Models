import os
import pathlib
from stage import measure
def get_filename(path: str) -> str:
    return os.path.split(path)[1].split('.')[0]

def is_newer_than(reference, file) -> bool:
    if not os.path.isfile(file):
        print(f'No file {file} found')
        return False
    else:
        return os.stat(reference).st_mtime <= os.stat(file).st_mtime 

def read_lines(filepath) -> list[str]:
    with open( filepath , "r" ) as read:
        text = read.read().splitlines()
    return text

def load_setfile(filename) -> set[str]: 
    return {get_filename(s) for s in read_lines(filename)}
