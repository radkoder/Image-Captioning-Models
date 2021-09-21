
from typing import Any, AnyStr, List, Tuple
IntList2D = List[List[int]]
IntList = List[int]
def flatten(t : List[List[Any]]) -> List[Any]:
    return [item for sublist in t for item in sublist]

def make_divs(arr:List[Any]) -> List[Tuple[List[Any],List[Any]]]:
    A,B = [],[]
    for i in range(1,len(arr)):
        a,b = arr[:i],arr[i]
        A.append(a)
        B.append(b)
    return A,B

def max_len(arr: List[Any]) -> int:
    return max(len(d) for d in arr)
def unzip_xy_pairs(arr:List[Tuple[ IntList2D, IntList ]]) -> Tuple[IntList2D, IntList]:
    return flatten([s[0] for s in arr]), flatten([s[1] for s in arr])
def peek(arr,n=3):
    for i in range(n):
        print(f'{i}: {arr[i]}')

import threading
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.

     A generic iterator and generator that takes any iterator and wrap it to make it thread safe.
    This method was introducted by Anand Chitipothu in http://anandology.com/blog/using-iterators-and-generators/
    but was not compatible with python 3. This modified version is now compatible and works both in python 2.8 and 3.0 
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

