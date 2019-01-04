#!/bin/bash

# PoS tagging task
python3 sequence_tagger.py --rnn_dim 512 --lr .1 --use_l_m 1 --use_word_emb 1 --use_char_emb 1 --use_words 1 --patience 5 --epochs 50 \
--batch_size 32 --eval_batch_size 32 --use_crf 1 --bn_sent 1 --bn_char 0 --char_emb_type rnn --task pos


# NER task
python3 sequence_tagger.py --rnn_dim 256 --lr .1 --use_l_m 1 --use_word_emb 1 --use_char_emb 1 --use_words 1 --patience 5 --epochs 35 \
--batch_size 32 --eval_batch_size 32 --use_crf 1 --bn_sent 1 --bn_char 0 --char_emb_type rnn --task ner

# MWE task
python3 sequence_tagger.py --rnn_dim 512 --lr .1 --use_l_m 1 --use_word_emb 1 --use_char_emb 1 --use_words 1 --patience 5 --epochs 20 \
--batch_size 32 --eval_batch_size 32 --use_crf 1 --bn_sent 1 --bn_char 1 --char_emb_type cnn --cnne_filters 300 --cnne_max 4 --task mwe