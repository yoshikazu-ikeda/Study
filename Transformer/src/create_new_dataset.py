import re
import glob

from path_schema import *

file_path = "//fukuoka/share/YoshikazuIkeda/Transformer/data/capdata"


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


files = sorted(glob.glob(file_path + '/*'), key=natural_keys)

# for file in files:
#     print(file)
i = 0
with open(file_path + '/new_beh.csv', 'a') as f:
    for file in files:
        with open(file, 'r') as g:
            f.write(g.read())
        print(i, '番目のファイル書き込み完了')
        i += 1
