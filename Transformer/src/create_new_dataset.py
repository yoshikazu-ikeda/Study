import re
import glob

import numpy as np
import csv
from path_schema import *

file_path = f"{DATA_PATH}"


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


files = sorted(glob.glob(file_path + '/Beh/*'), key=natural_keys)

# for file in files:
#     print(file)
i=0
with open(file_path + '/new_beh.csv', 'a') as f:
    for file in files:
        with open(file, 'r') as g:
            f.write(g.read())
        print(i,'番目のファイル書き込み完了')
        i+=1
