import os,zipfile,shutil
from typing import *

def get_filename(path: str) -> str:
    return os.path.split(path)[1].split('.')[0]

def is_newer_than(reference, file) -> bool:
    if not os.path.isfile(file):
        print(f'No file {file} found')
        return False
    else:
        return os.stat(reference).st_mtime <= os.stat(file).st_mtime 

def read(filepath: str) -> str:
    print(f"Reading file: {os.path.abspath(filepath)}")
    with open( filepath , "r" ) as read:
        text = read.read()
    return text

def read_lines(filepath: str) -> List[str]:
    print(f"Reading file: {os.path.abspath(filepath)}")
    with open( filepath , "r" ) as read:
        text = read.read().splitlines()
    return text

def write(filepath: str, content: str):
    filepath = os.path.abspath(filepath)
    print(f"Writing to file: {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath,"w") as w:
        w.write(content)

def load_setfile(filename: str) -> Set[str]:
    print(f'Reading setfile: {os.path.abspath(filename)}')
    return {get_filename(s) for s in read_lines(filename)}

def unpack(zippath: str,files: List[str]) -> None:
    zippath = os.path.abspath(zippath)
    with zipfile.ZipFile(zippath) as z:
        dest = os.path.dirname(os.path.abspath(zippath))
        for f in files:
            print(f'Unpacking {f} from {zippath} to {dest}')
            z.extract(f,dest)

def delete(files: List[str]) -> None:
    for f in files:
        print(f'Removing file {os.path.abspath(f)}')
        os.remove(f)

def move(src, dst):
    print(f"Moving file {src} to {dst}")
    shutil.move(src,dst)
    
def file(name):
    os.makedirs(os.path.dirname(name), exist_ok=True)
    return open(name,"w")

