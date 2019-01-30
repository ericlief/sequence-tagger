# sequence-tagger

This is a multipurpose sequence tagger which grew out of my master's thesis ("Deep contextualized word embeddings from character language models for neural sequence labeling," Institute of Formal and Applied Linguistics, Charles University).

Abstract: A family of Natural Language Processing (NLP) tasks such as part-of-speech (PoS) tagging, Named Entity Recognition (NER), and Multiword Expression (MWE) identification all involve assigning labels to sequences of words in text (sequence labeling).  Most modern machine learning approaches to sequence labeling utilize word embeddings, learned representations of text, in which words with similar meanings have similar representations. Quite recently, contextualized word embeddings have garnered much attention because, unlike pretrained context-insensitive embeddings such as word2vec, they are able to capture word meaning in context. In this thesis, I evaluate the performance of different embedding setups (context-sensitive, context-insensitive word, as well as task-specific word, character, lemma, and PoS) on the three abovementioned sequence labeling tasks using a deep learning model (BiLSTM) and Portuguese datasets.is customizable for most common sequence labeling tasks such as part-of-speech (PoS) tagging, named entity recognition (NER), and multiword expression (MWE) identificition.  

For optimal results the tagger can be used with a pretrained language model such as Flair (Akbik et al. 2018), in combination with both pretrained and task-specific embeddings. 

For a demo, you can use the included script run.sh for these tasks using a pretrained Flair (Akbik et al. 2018) Portuguese character language model (CharLM) trained on close to 1B words of CommonCrawl text. You can also train your own LM and use this instead.

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
