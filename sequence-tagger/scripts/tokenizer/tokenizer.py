# -*- coding: utf-8 -*-

import regex as re    


    
def tokenize(text, token_min_len=1, token_max_len=50, lower=False):
    """
    Tokenize the given sentence in Portuguese.
    :param text: text to be tokenized, as a string
    :param token_min_len: int
    :param token_max_len: int
    :param lower: lowercase, bool)
    
    """
    abbrev_fh = '/home/liefe/mypkgs/mypkgs/tokenizer/abbrev/abbreviations_pt.py'
        

    # Read abbrev list and build regex for these
    abbrev = ""    
    try:
        with open(abbrev_fh, 'r') as f:
            for line in f:
                pre = line.strip()
                if pre and pre[0] != '#':
                    pre = re.escape(pre, special_only=True, literal_spaces=True)
                    abbrev += pre + '|'
                    
    except IOError:
        print('Abbreviation file not found')
    
    
    
    # Turn ` into '
    text = re.sub('`', "'", text)
    text = re.sub("''", '"', text)
    
    # Get rid of extraneous spaces
    text = re.sub(" {2,}", " ", text)
    

    
    regex = r"""(?uxi) # Flags: UNICODE, VERBOSE, IGNORECASE
    
    # Numbers
    #(?:\(?\+?\d{1,3}\)?)?\s*(?:\d+[\.\s-]?)+\d+|
    #(?:\(?\+?\d{1,3}\)? )?\d+(?:[,\. -]\d+)*|
    (?:\(?\+?\d{1,3}\)?[ ]*)?\d+(?:[:\/,\. -]\d+)*|  
    
    
    # Numbers in format 999.999.999,999, and (216) 729-9295, 4:15
    # possibly followed by hyphen and alphanumerics, \B- avoids A-1.1-a-a producing a negative number
    #(?:\B-)?\d+(?:[,\.]\d+)*(?:-?\w)*| 
    #\d+(?:[,\.]\d+)*| 
    
    
    # One letter abbreviations, e.g. E.U.A.  
    (?:\w\.)+| 
    
    # Abbreviations/Nonbreaking prefixes from list
    #(?:\b # doesn't seem to work a/c/ -> a/c /, on one line throws error for extra )
    %s
    #\b)
    #|

    # Emails
    #(?:[a-z0-9!#$%%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%%\&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])|
    
    # URLs
    #(?:https?://)?\w{2,}(?:\.\w{2,})+(?:/\w+)*|   
    
    # Hashtags and twitter names
    #(?:[\#@]\w+)| 
      
    ## Phone numbers
    ##(?:\(?\+?\d{1,3}\)?)?\s*(?:\d+[\.\s-]?)+\d+|

    # Ellipsis or sequences of dots
    \.{3,}|                           

    # Alphanumerics
    \w+|                              

    # Any sequence of dashes 
    -+|

    # Any non-space character
    \S  
    
    """ % abbrev
    
  
    p = re.compile(regex)
    return p.findall(text)

   

if __name__ == "__main__":

    import lzma
    
    s = r"E.U.A., Il.mo  a/c a/c/ c/ ele. pág. 2, sec. 2.3.1 Sup... 12:12:23 12/2/2012 \'\`  10,10,10 33-33-33 100.0.1,33 ''Dr. Erique A. Lief, Ph.D'', dra. Rocha, il.mo sr. Moraes. telefone +420 77.655.65.49 +420 77 655 65 49 e ele... a/c Ilma./ilma./il.ma/Ilma N.Sra. Garcia' . esse grande cabrão!--que é Dra. Morções; seu João às costas! 10,00.34, '4:15 p.m.' eric23_lief28@seznam.cz"

    
    print(tokenize(s))
    
