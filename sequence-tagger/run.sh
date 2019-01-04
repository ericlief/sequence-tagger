#!/bin/bash


python3 sequence_tagger.py --rnn_dim 256 --lr .1 --use_l_m 1 --use_word_emb 1 --use_char_emb 1 --use_words 1 --patience 20 --epochs 1 \
--batch_size 32 --eval_batch_size 32 --use_crf 1 --downsize_lm 0 --bn_sent 1  --bn_char 1 \
--word_dropout 0 --locked_dropout 0 --char_emb_type cnn --task pos