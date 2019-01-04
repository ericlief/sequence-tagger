import re
from collections import Counter

fh = "/home/liefe/data/pt/mwe/mwe10/dev.parsemetsv"
cnts = Counter()
freqs = {}

with open(fh, 'rt') as f:
    for line in f:
        #feats = line.split()
        #mwe = feats[3]
        mwes = re.findall("[0-9]:[a-zA-Z]+", line)
        #print(mwes)
        #line = re.sub("[0-9]:([a-zA-Z]+)", "\1", line) 
        #mwes = re.findall("[a-zA-Z]+", line)
        if mwes:
            for mwe in mwes:
                id, item = mwe.split(':')
                cnts[item] += 1
                

n = float(sum(cnts.values()))
for k,v in cnts.items():
    freqs[k] = v / n

print(n)
print(cnts)
print(freqs)