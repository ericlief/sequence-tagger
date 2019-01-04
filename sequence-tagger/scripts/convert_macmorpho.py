"""
Script to convert Mac-Morpho format into column format
Word Tag
"""

def process(fname):
    
    words_tags = []
    with open(fname, "rt") as f:
        for line in f:
            line = line.split()
            words_tags.append(line)
            
    with open(fname[:-3]+"cols.txt", "wt") as f:
        for line in words_tags:
            for word_tag in line:
                word, tag = word_tag.split("_")
                print(word, tag, file=f) 
            print("", file=f)
        
if __name__ == "__main__":
    process("data/macmorpho/train.txt")
    process("data/macmorpho/dev.txt")
    process("data/macmorpho/test.txt")

        


