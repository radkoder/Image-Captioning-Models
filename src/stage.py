import time, functools
from math import floor
def stostr(sec: int) -> str:
    h = sec/3600
    hfin = floor(h)

    min = (h-hfin)*60
    minfin = floor(min)

    s = (min - minfin)*60
    sfin = floor(s)
    return f'{hfin}h {minfin}min {sfin}s'


class ProgressBar:
    def __init__(self, title, max) -> None:
        self.barlen = 10
        self.title = title
        self.max = max
        self.count = 0
        self.meanTime = 0
        self.totalTime = 0
        self.printQueue = []
        self.lastTime = 0
        self.currTime = time.time()

    def update(self,status) -> None:
        self.lastTime = self.currTime
        self.currTime = time.time()
        self.count += 1
        self.print_bar(status)
    def print_bar(self,status) -> None:
        r = int(self.count*self.barlen/self.max)
        s = '.'*r + ' '*(self.barlen-r)
        diff = self.currTime - self.lastTime
        outstr = f'{self.title}:[{s}][{self.count}/{self.max}] => {status} [{stostr(int(diff))}][eta: {stostr(int(diff*self.max))}]'
        print(outstr+'   ',end='\r')
    def end(self) -> None:
        self.print_bar("DONE")
        print('')


def measure(name):
    def decorator(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            print(name + "...")
            begin = time.time()
            ret = func(*args, **kwargs)
            end = time.time()
            print(name+f' done in {end-begin:.2f} s')
            return ret
        return inner
    return decorator