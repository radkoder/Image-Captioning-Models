import os,zipfile
def get_filename(path: str) -> str:
    return os.path.split(path)[1].split('.')[0]

def is_newer_than(reference, file) -> bool:
    if not os.path.isfile(file):
        print(f'No file {file} found')
        return False
    else:
        return os.stat(reference).st_mtime <= os.stat(file).st_mtime 

def read_lines(filepath: str) -> list[str]:
    with open( filepath , "r" ) as read:
        text = read.read().splitlines()
    return text

def load_setfile(filename: str) -> set[str]:
    return {get_filename(s) for s in read_lines(filename)}

def unpack(zippath: str,files: list[str]) -> None:
    with zipfile.ZipFile(zippath) as z:
        dest = os.path.dirname(os.path.abspath(zippath))
        for f in files:
            print(f'Unpacking {f} from {zippath} to {dest}')
            z.extract(f,dest)

def delete(files: list[str]) -> None:
    for f in files:
        print(f'Removing file {f}')
        os.remove(f)