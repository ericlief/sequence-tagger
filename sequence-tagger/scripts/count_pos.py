import re
from collections import Counter
import sys

fh = sys.argv[1]
cnts = Counter()
freqs = {}

with open(fh, 'rt') as f:
    for line in f:
        #feats = line.split()
        #mwe = feats[3]
        #tags = re.findall("B-[A-Z]+", line)
        #print(mwes)
        #line = re.sub("[0-9]:([a-zA-Z]+)", "\1", line) 
        #mwes = re.findall("[a-zA-Z]+", line)
        if line[0] != "\n":
            word, tag = line.split()
            cnts[tag] += 1

n = float(sum(cnts.values()))
for k,v in cnts.items():
    freqs[k] = v / n

print(n)
print(cnts)
print(freqs)