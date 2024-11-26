#!/usr/bin/env python3
import time
class screen:
    def __init__(self):
        print("\n")
        self.strings = []
        self.cursor_pos = (0,0)

    def clear(self):
        print(f"\033[{len(self)-self.cursor_pos[1]-1}A"+("\033[2K\033[1B"*(len(self)+2))+f"\033[{len(self)}A",end='\r')
        self.cursor_pos = (0,0)

    def show(self):
        print('\n'.join(self.strings),end="")
        self.cursor_pos = (0,len(self.strings)-1)

    def append(self, string):
        self.strings.append(str(string))
        self.clear()
        self.show()

    def __len__(self):
        return len(self.strings)

    def __getitem__(self, i):
        return self.strings[i]

    def __setitem__(self, i, v):
        self.clear()
        self.strings[i] = str(v)
        self.show()


class Status:
    def __init__(self, title=None):
        self.data = 0
        self.title = f"{title}: " if not title is None else ''
        self.offset = len(self.title)

        LOG_SCREEN.append(f"{self.title}{self.data}")
        self.pos = len(LOG_SCREEN.strings)-1

    def show(self):
        LOG_SCREEN[self.pos] = f"{self.title}{self.data}"

    def update(self, data):
        self.data = data
        self.show()

    def bar(self, bar_size=30):
        return bar(title=self.title, bar_size=bar_size, pos=self.pos)


def log_action(message):
    def decorated(function):
        def new_f(*args, **kwargs):
            parsed_message = message.format(*args, **kwargs)
            LOG_SCREEN.append(parsed_message)
            pos = len(LOG_SCREEN.strings)-1
            if DEBUG:
                print(f" ({args =}, {kwargs =})", end='')

            out = function(*args, **kwargs)
            LOG_SCREEN[pos] = parsed_message + ": Done"
            return out
        return new_f
    return decorated


def bar(title=None, bar_size=30, pos=None):
    idx = pos
    def wrap(func):
        idx = pos
        if not title is None:
            txt = f"{title}: "
        else:
            txt = ''
        offset = len(txt)
        def f(iterator):
            idx = pos
            if idx is None:
                LOG_SCREEN.append(f"{txt}[{'-'*bar_size}]")
                idx = len(LOG_SCREEN.strings)-1
            else:
                LOG_SCREEN[idx] = f"{txt}[{'-'*bar_size}]"

            for i,item in enumerate(iterator):
                p_last = offset+round(((i-1)/len(iterator))*bar_size)
                p = offset+round((i/len(iterator))*bar_size)
                filled = ("#"*(p-p_last))
                LOG_SCREEN[idx] = LOG_SCREEN[idx][:p+1]+filled+LOG_SCREEN[idx][p+len(filled)+1:]
                yield i, item, func(*item)
        return f

    return wrap

#bar("test")(time.sleep)([0.1]*30)

LOG_SCREEN = screen()
DEBUG = False

if __name__ == "__main__":
    print("\n")
    test = screen()
    LOG_SCREEN.append("0")
    LOG_SCREEN.append("1")
    LOG_SCREEN.append("2")
    LOG_SCREEN.append("3")
    LOG_SCREEN.append("4")
    LOG_SCREEN.append("5")
    LOG_SCREEN.append("6")
    LOG_SCREEN.append("7")

    #for i in range(30):
    #    a = test[3][:i+1]+'#'+test[-1][i+2:]
    #    test[3] = a
    #    time.sleep(0.1)
    #
    #
    for x,_,_ in bar(title="test",pos=0)(time.sleep)([0.1]*30):
        for y,_,_ in bar(pos=1)(time.sleep)([0.1]*10):
            pass
