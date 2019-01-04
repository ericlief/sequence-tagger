"""
Script to convert PARSEME format to IOBE/BIOSE
Usage python convert_mwe_to_iobe.py input-file
"""

if __name__ == "__main__":
    
    fh = sys.args[1]
    
    with open(fh, "rt") as fin:
        lines = fin.readlines()
        prev_tag = None
        for i, line in enumerate(lines):
            if len(line) > 1 and line[0] != "#":
                cols1 = line.split()
                if len(cols1) != 11:
                    raise Exception("conll format incorrect")
                if cols1[10] == "*" or cols1[10][0:2] in ["I-", "E-"]:
                    #print("already processed", cols[10])
                    continue
                mwes = cols1[10].split(";")
                mwe = mwes[0].split(":") # check first mwe to see if we can throw out second/embedded mwe
                
                #print(len(mwe), mwe)
                if len(mwe) == 1: # just an id, e.g. 2 or 2;3 or 2;3:LVC.full
                    if len(mwes) > 1 and len(mwes[1]) > 1: # more than one tag 2;3:LVC.full
                        #print(len(mwes))
                        #if :                               
                        mwe = mwes[1].split(":")   
                    else: # e.g. 2 or 2;3 
                        print("\n\nerasing ", i, mwes, cols1[10])
                        print(lines[i])
                        
                        cols1[10] = "*\n" # erase remains of thrown out tag
                        lines[i] = "\t".join(cols1) # transform line to BIOES format
                        print(lines[i])
                        continue
                  
                id, type = mwe  
                last_offset = None
                for j, next_line in enumerate(lines[i+1:]): # forward search for continuing tag
                    if len(next_line) == 1: # end of sentence stop
                        break
                    cols2 = next_line.split()
   
                    if len(cols2) != 11:
                        raise Exception("conll format incorrect")                    
                    if cols2[10] == "*":  # skip all *'s 
                        continue
                    
                    if id in cols2[10]: # will throw out the second tag in annotations like this 1;2:
                       
                        last_offset = i + j + 1                        
                        cols2[10] = "I-" + type + "\n"
                        lines[last_offset] = "\t".join(cols2)  # transform line to BIOES format
                
                if last_offset:
                    cols1[10] = "B-" + type + "\n"
                    print("B replaced", cols1[10], i)
                    lines[i] = "\t".join(cols1) # transform line to BIOES format
                    
                    cols = lines[last_offset].split()
                    cols[10] = "E-" + type + "\n"
                    lines[last_offset] = "\t".join(cols)
     
                else:
                    print("removing line ", i, cols1[10], lines[i])
                    cols1[10] = "*\n"
                    lines[i] = "\t".join(cols1) # transform line to BIOES format
                                    
    with open("data/mwe/train.txt", "wt") as fout:
        for line in lines:
            fout.write(line)
             
     