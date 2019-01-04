
if __name__ == "__main__":
    
    fh_mwe = "/home/liefe/data/pt/mwe/mwe10/train.mwe"
    fh_conll = "/home/liefe/data/pt/mwe/mwe10/train.conllu"
    fh_out = "/home/liefe/data/pt/mwe/mwe10/train.txt"
    
    with open(fh_mwe, "rt") as f_mwe:   
        with open(fh_conll, "rt") as f_conll:
            with open(fh_out, "wt") as f_out:
                
                mwe_lines = f_mwe.readlines()
                conll_lines = f_conll.readlines()
                assert len(mwe_lines) == len(conll_lines)
                for i in range(len(mwe_lines)):
                    
                    feats = mwe_lines[i].split()
                    print(feats)
                    if len(feats) == 4:
                        mwe = feats[3]
                    else:
                        print("", file=f_out)
                        continue
                    print(conll_lines[i].rstrip() + "\t" + mwe, file=f_out)
                