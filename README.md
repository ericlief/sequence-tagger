# sequence-tagger

This is a multipurpose sequence tagger which is customizable for most common sequence labeling tasks such as part-of-speech (PoS) tagging, named entity recognition (NER), and multiword expression (MWE) identificition.  

For a demo, you can use the included script run.sh for these tasks using a pretrained Flair (Akbik et al. 2018) Portuguese character language model (CharLM) trained on close to 1B words of CommonCrawl text.  

Dependencies:  

tensorflow-gpu  
pip3 install tensorflow-gpu

flair  
pip3 install flair 

DEMO:  
./run.sh  

There are numerous hyperparameters. For example, one can choose character-level embedddings via RNN or CNN and specify the parameters to use.

For instance, the run.sh script will run the PoS tagger with the following parameters:

# PoS tagging task
python3 sequence_tagger.py --rnn_dim 512 --lr .1 --use_l_m 1 --use_word_emb 1 --use_char_emb 1 --use_words 1 --patience 5 --epochs 50 --batch_size 32 --eval_batch_size 32 --use_crf 1 --bn_sent 1 --bn_char 0 --char_emb_type rnn --task pos

The results are state of the art (95.49%) accuracy.  

For evaluation, use the included CoNLL-2003 evaluation script.  
Note that output will be in a time stamped directory with the parameters used in the logs directory

Use the -r ('raw') flag for non per entity evaluation (PoS tagging).  
./conlleval -r < [output] 

For NER, we want per MWE metrics:  

./conlleval < [output] 
