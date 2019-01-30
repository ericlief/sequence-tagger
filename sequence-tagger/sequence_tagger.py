#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from collections import deque
import random
import torch
from torch._six import inf
from collections import defaultdict


class SequenceTagger:
    def __init__(self, 
                 args,
                 corpus,
                 seed=42,
                 lm=None,
                 word_emb=None,
                 restore_model=False,
                 model_path=None,
                 char_dict_path=None):  
      
        
        self.corpus = corpus # flair corpus type: List of Sentences  
        self.tag_type = args.task # what we're predicting
        self.metrics = Metrics() # for logging metrics
        self.lm = lm  # the language model
        self.word_emb = word_emb  # the pretrained word emb
        
        # Instantiate scheduler for learning rate annealing
        self.scheduler = ReduceLROnPlateau(args.lr, args.annealing_factor, args.patience) 
        
        # Make the tag dictionary for labels from the corpus
        self.tag_dict = corpus.make_tag_dictionary(tag_type=self.tag_type)  # id to tag
        n_tags = len(self.tag_dict)
         
        # Get words in corpus for in task word embeddings 
        if args.use_words:
            #train_data = corpus.train
            data = corpus.train + corpus.dev + corpus.test
            self.word_dict = {'pad': 0, 'unk': 1}
            self.words = ['pad', 'unk']        
            for i in range(len(data)):
                for j in range(len(data[i])): 
                    word = data[i][j].text
                    if word not in self.word_dict:
                        self.word_dict[word] = len(self.words)
                        self.words.append(word)   
        
        # Make dictionary for pos tags and lemmas if desired       
        if args.use_lemmas:
            self.lemma_dict = corpus.make_tag_dictionary("lemma")  # id to tag
        if args.use_pos_tags:
            self.pos_tag_dict = corpus.make_tag_dictionary("upos")  # id to tag
            
        
        # Create graph and session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(log_device_placement=False))
        
        # Construct graph
        self.construct(args, n_tags, lm, word_emb, restore_model, model_path, char_dict_path)
        
    def construct(self, args, n_tags, lm, word_emb, restore_model, model_path, char_dict_path):
        
        with self.session.graph.as_default():

            
            # Shape = (batch_size, max_sent_len)
            self.gold_tags = tf.placeholder(tf.int32, [None, None], name="tags")            
            # Trainable params or not
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")
            # Shape = (batch_size)
            self.sentence_lens = tf.placeholder(tf.int32, [None], name="sentence_lens")
            
            inputs = []  # concatenated inputs
            
            # Pretrained CharLM embeddings
            if args.use_l_m:
                self.lm_emb_dim = lm.embedding_length                
                # Shape = (batch_size, max_sent_len, lm_emb_dim)
                self.lm_emb = tf.placeholder(tf.float32, [None, None, self.lm_emb_dim], name="lm_emb")            
                                
                # Downsize lm if desired. This will reduce the dimensionality of the lm emb and 
                # potentially could help with overfitting
                lm_emb = self.lm_emb
                if args.downsize_lm:
                    new_dim = args.downsize_lm * self.lm_emb_dim
                    if args.bn_input:
                        lm_emb = tf.layers.batch_normalization(lm_emb, training=self.is_training, name="bn_input")
                    elif args.dropout_input:
                        lm_emb = tf.nn.dropout(lm_emb, 1 - args.dropout_input, name="dropout_input")
                    lm_emb_reduced = tf.layers.dense(lm_emb, new_dim, name="charlm_input_layer")
                    inputs.append(lm_emb_reduced)
                else:
                    inputs.append(lm_emb)
            
            # Use pretrained word emb         
            if args.use_word_emb:
                self.word_emb_dim = self.word_emb.vector_size  
                # Shape = (batch_size, max_sent_len, word_emb_dim)                                
                self.emb_words = tf.placeholder(tf.float32, [None, None, self.word_emb_dim], name="lm_emb")                            
                inputs.append(self.emb_words) 
            
            # Use in task word emb    
            if args.use_words:
                # Shape = (batch_size, max_sent_len)                
                self.word_ids = tf.placeholder(tf.int32, [None, None], name="word_ids")
                n_words = len(self.word_dict)
                self.word_embedding = tf.get_variable("word_embedding", [n_words, args.rnn_dim], tf.float32)
                # Shape = (batch_size, max_sent_len, rnn_dim)                
                embedded_words = tf.nn.embedding_lookup(self.word_embedding, self.word_ids)
                inputs.append(embedded_words)
            
            # Use in task lemma emb
            if args.use_lemmas:
                # Shape = (batch_size, max_sent_len)                
                self.lemma_ids = tf.placeholder(tf.int32, [None, None], name="lemmas")
                n_lemmas = len(self.lemma_dict)
                self.lemma_embedding = tf.get_variable("lemma_embedding", [n_lemmas, args.rnn_dim], tf.float32)
                # Shape = (batch_size, max_sent_len, rnn_dim)                
                embedded_lemmas = tf.nn.embedding_lookup(self.lemma_embedding, self.lemma_ids)
                inputs.append(embedded_lemmas)
            
            # Use in task pos emb
            if args.use_pos_tags:
                # Shape = (batch_size, max_sent_len)                
                self.pos_tag_ids = tf.placeholder(tf.int32, [None, None], name="tags")
                n_pos_tags = len(self.pos_tag_dict)
                self.pos_tag_embedding = tf.get_variable("tag_embedding", [n_pos_tags, args.rnn_dim], tf.float32)
                # Shape = (batch_size, max_sent_len, rnn_dim)                
                embedded_pos_tags = tf.nn.embedding_lookup(self.pos_tag_embedding, self.pos_tag_ids)
                inputs.append(embedded_pos_tags)
                
            # Use character-level (cle) embeddings
            if args.use_char_emb:
                # Shape = (batch_size, max_sent_len)
                self.char_seq_ids = tf.placeholder(tf.int32, [None, None], name="char_seq_ids")
                # Shape = (num_char_seqs, indef]
                self.char_seqs = tf.placeholder(tf.int32, [None, None], name="char_seqs")
                # Shape = (batch_size)
                self.char_seq_lens = tf.placeholder(tf.int32, [None], name="char_seq_lens")                
                with tf.variable_scope('cle'):
                   
                    # Build char dictionary to map chars to indices
                    if char_dict_path is None:
                        self.char_dict = Dictionary.load('common-chars')
                    else:
                        self.char_dict = Dictionary.load_from_file(cle_path)    
                    num_chars = len(self.char_dict.idx2item)
                    
         
                    # Generate character embeddings for num_chars of dimensionality cle_dim.
                    self.char_embeddings = tf.get_variable('char_embeddings', [num_chars, args.cle_dim])
                    
                    # Embed self.chaseqs (list of unique words in the batch) using the character embeddings.
                    embedded_chars = tf.nn.embedding_lookup(self.char_embeddings, self.char_seqs)
                    
                    # Use rnn for char emb
                    if args.char_emb_type == "rnn":
                        cell_fw = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(256)
                        cell_bw = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(256)
                        
                        # Dropout wrapper
                        if args.locked_dropout:
                            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_keep_prob=1-args.locked_dropout, output_keep_prob=1-args.locked_dropout)
                            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, input_keep_prob=1-args.locked_dropout, output_keep_prob=1-args.locked_dropout)
                        
                        # BN and dropout for normalization   
                        if args.bn_char:
                            embedded_chars = tf.layers.batch_normalization(embedded_chars, training=self.is_training, name="bn_char")
                        if args.dropout_char:
                            embedded_chars = tf.nn.dropout(embedded_chars, 1 - args.dropout_char, name="dropout_char") 
                            
                        # Run cell in limited scope fw and bw
                        # in order to encode char information (subword factors)
                        # The cell is size of char dim, e.g. 32 -> output (?,32)
                        _, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                                    cell_bw, 
                                                                    inputs=embedded_chars, 
                                                                    sequence_length=self.char_seq_lens, 
                                                                    dtype=tf.float32)
                        
                        # Sum the resulting fwd and bwd state to generate character-level word embedding (CLE)
                        # of unique words in the batch                   
                        cle = tf.reduce_sum(states, axis=0) 
                        # Shape = (batch_size, max_sent_len, cle_dim)                                    
                        embedded_char_seqs = tf.nn.embedding_lookup(cle, self.char_seq_ids)
                        inputs.append(embedded_char_seqs)
            
                        
                    # Use cnn for char emb
                    else:
                        # For kernel sizes of {2..args.cnne_max}, do the following:
                        # - use `tf.layers.conv1d` on input embedded characters, with given kernel size
                        #   and `args.cnne_filters`; use `VALID` padding, stride 1 and no activation.
                        # - perform channel-wise max-pooling over the whole word, generating output
                        #   of size `args.cnne_filters` for every word.
                        features = []
                        for kernel_size in range(2, args.cnne_max + 1):
                            conv = tf.layers.conv1d(embedded_chars, args.cnne_filters, kernel_size, strides=1, padding="VALID", activation=None, use_bias=False, name="cnne_layer_"+str(kernel_size))
                            
                            # Apply BN or dropout
                            if args.bn_char:
                                conv = tf.layers.batch_normalization(conv, training=self.is_training, name="bn_cnn"+str(kernel_size))
                            if args.dropout_char:
                                conv = tf.nn.dropout(conv, 1 - args.dropout_char, name="dropout_char"+str(kernel_size)) 
                            
                            pooling = tf.reduce_max(conv, axis=1)
                            features.append(pooling)
    
    
                        # Concatenate the computed features (in the order of kernel sizes 2..args.cnne_max).
                        # Consequently, each word from `self.charseqs` is represented using convolutional embedding
                        # (CNNE) of size `(args.cnne_max-1)*args.cnne_filters`.                
                        features_concat = tf.concat(features, axis=1)
                        # Generate CNNEs of all words in the batch by indexing the just computed embeddings
                        # by self.charseq_ids (using tf.nn.embedding_lookup).
                        cnne = tf.nn.embedding_lookup(features_concat, self.char_seq_ids)
                        # Concatenate the word embeddings (computed above) and the CNNE (in this order).
                        #inputs = tf.concat([inputs, cnne], axis=-1)                          
                        
                        inputs.append(cnne)
                                        
                    
            # Concatenate all embeddings
            if len(inputs) > 1:
                inputs_concat = tf.concat(inputs, axis=-1)      
            elif len(inputs) == 1:
                inputs_concat = inputs[0]
            else:
                raise Exception("There must be at least one type of input embedding")
            
            # Normalization
            if args.bn_sent:
                inputs_concat = tf.layers.batch_normalization(inputs_concat, training=self.is_training, name="bn_sent")
            elif args.dropout_sent:
                inputs_concat = tf.nn.dropout(inputs_concat, 1 - args.dropout_sent, name="dropout_sent")
            
            # Apply word dropout
            if args.word_dropout:
                inputs_concat = self.word_dropout(inputs_concat, args.word_dropout)
                    
            # Choose RNN cell according to args.rnn_cell (LSTM and GRU)
            if args.rnn_cell == 'GRU':
                cell_fw = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(args.rnn_dim)
                cell_bw = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(args.rnn_dim)

            elif args.rnn_cell == 'LSTM':
                cell_fw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(args.rnn_dim)
                cell_bw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(args.rnn_dim)

            else: 
                raise Exception("Must select an rnn cell type")     
 
            # Add locked/variational dropout wrapper
            # Warning this does not yield favorable results
            if args.locked_dropout:
                cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_keep_prob=1-args.locked_dropout, output_keep_prob=1-args.locked_dropout, state_keep_prob=1, variational_recurrent=True, input_size=inputs_concat.get_shape()[-1], dtype=tf.float32)
                cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, input_keep_prob=1-args.locked_dropout, output_keep_prob=1-args.locked_dropout, state_keep_prob=1, variational_recurrent=True, input_size=inputs_concat.get_shape()[-1], dtype=tf.float32)

            # Process embedded inputs with rnn cell
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                         cell_bw, 
                                                         inputs_concat, 
                                                         sequence_length=self.sentence_lens, 
                                                         dtype=tf.float32,
                                                         time_major=False)
           
                            
            # Concatenate the outputs for fwd and bwd directions (in the third dimension).
            outputs_concat = tf.concat(outputs, axis=-1)
            
            # Normalization
            if args.bn_output:
                outputs_concat = tf.layers.batch_normalization(outputs_concat, training=self.is_training, name="bn_output")
           
            if args.dropout_output:
                outputs_concat = tf.nn.dropout(outputs_concat, 1 - args.dropout_output, name="dropout_output")
 
            
            # Add a dense layer (without activation) into num_tags classes 
            logits = tf.layers.dense(outputs_concat, n_tags) 

            # Decoding
            
            # Generate `weights` as a 1./0. mask of valid/invalid words (using `tf.sequence_mask`).
            weights = tf.sequence_mask(self.sentence_lens, dtype=tf.float32)

            # Use crf for decoding
            if args.use_crf:

                # Compute log likelihood and transition parameters using tf.contrib.crf.crf_log_likelihood
                # and store the mean of sentence losses into `loss`.
                self.transition_params = tf.get_variable("transition_params", [n_tags, n_tags], initializer=tf.glorot_uniform_initializer())
                log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(inputs=logits, 
                                                                                           tag_indices=self.gold_tags, 
                                                                                           sequence_lengths=self.sentence_lens,
                                                                                           transition_params=self.transition_params)

                self.loss = tf.reduce_mean(-log_likelihood)
                self.reduc_loss = self.loss
                # Compute the CRF predictions into `self.predictions` with `crf_decode`.
                self.predictions, self.scores = tf.contrib.crf.crf_decode(logits, self.transition_params, self.sentence_lens)

            # Use local softmax decoding
            else:             
                self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.gold_tags, logits=logits, weights=weights)
                self.reduc_loss = tf.reduce_mean(self.loss)  

                # Generate `self.predictions`.
                self.predictions = tf.argmax(logits, axis=-1)              

            global_step = tf.train.create_global_step()            
  
           
            # Choose optimizer
            if args.optimizer == "SGD":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.scheduler.lr) 
            else:                
                optimizer = tf.train.AdamOptimizer(learning_rate=self.scheduler.lr) 

            # Note how instead of `optimizer.minimize` first get the # gradients using
            # `optimizer.compute_gradients`, then optionally clip them and
            # finally apply then using `optimizer.apply_gradients`.
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            # Compute norm of gradients using `tf.global_norm` into `gradient_norm`.
            gradient_norm = tf.global_norm(gradients) 
            
            # If args.clip_gradient, clip gradients (back into `gradients`) using `tf.clip_by_global_norm`.            
            if args.clip_gradient:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=args.clip_gradient, use_norm=gradient_norm)
            
            # Update ops for bn
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):     
                self.training = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
                
            # Summaries
            #self.current_accuracy, self.update_accuracy = tf.metrics.accuracy(self.gold_tags, self.predictions, weights=weights)
            self.current_loss, self.update_loss = tf.metrics.mean(self.loss, weights=tf.reduce_sum(weights))
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.update_loss),
                                           tf.contrib.summary.scalar("train/gradient_norm", gradient_norm),
                                           tf.contrib.summary.scalar("train/accuracy", self.metrics.accuracy),
                                           tf.contrib.summary.scalar("train/precision", self.metrics.precision),
                                           tf.contrib.summary.scalar("train/recall", self.metrics.recall),
                                           tf.contrib.summary.scalar("train/f1", self.metrics.f1)]                                   

            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.current_loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.metrics.accuracy),
                                               tf.contrib.summary.scalar(dataset + "/precision", self.metrics.precision),
                                               tf.contrib.summary.scalar(dataset + "/recall", self.metrics.recall),
                                               tf.contrib.summary.scalar(dataset + "/f1", self.metrics.f1)]                                               
            # To save model
            self.saver = tf.train.Saver()
            
            # Restore model from checkpoint
            if restore_model:
                self.saver.restore(self.session, model_path)  
                print("Restoring model from ", model_path)
            else:
                self.session.run(tf.global_variables_initializer())                 # initialize variables
            
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)


    def train(self, 
              args,
              train_data,
              dataset_name = "train",
              embeddings_in_memory=False,
              checkpoint=False,
              train_with_dev=True,
              metric="accuracy"): 
                    
        # Reset batch metrics
        self.session.run(self.reset_metrics)
        
        # Train epochs
        #if cv:
            #train_data, test_data = self.cv(args.cv)
        #train_data = corpus.train  
        
        dev_score = 0
        for epoch in range(args.epochs):
            
            # Stop if lr gets to small
            if self.scheduler.lr < args.final_lr:
                print("Learning rate has become to small. Exiting training: lr=", self.scheduler.lr)
                break    
            
            # Shuffle data and form batches
            random.shuffle(train_data)
            batches = [train_data[i:i + args.batch_size] for i in range(0, len(train_data), args.batch_size)]        
            
            # To store metrics
            totals_per_tag = defaultdict(lambda: defaultdict(int))
            totals = defaultdict(int)            
            
            for batch_n, batch in enumerate(batches):
                
                # Sort batch and get lengths
                batch.sort(key=lambda i: len(i), reverse=True)
                
                # Remove super long sentences which may cause memory issues                     
                max_sent_len = len(batch[0])                                    
                while len(batch) > 1 and max_sent_len > 200:
                    #print("removing long sentence")
                    batch = batch[1:]
                    max_sent_len = len(batch[0])                    
                    
                sent_lens = [len(s.tokens) for s in batch]
                n_sents = len(sent_lens)     
                
                # Prepare embeddings, pad and embed sentences and tags
                
                # Embed sentences using Flair CharLM
                if args.use_l_m:
                    self.lm.embed(batch)            
                    lm_emb = np.zeros([n_sents, max_sent_len, self.lm_emb_dim])
               
                # For pretrained word embedding such as fastText    
                if args.use_word_emb:
                    word_emb = np.zeros([n_sents, max_sent_len, self.word_emb_dim])
                
                # Prepare ids placeholders for trainable in-task embeddings 
                if args.use_words: 
                    word_ids = np.zeros([n_sents, max_sent_len]) 
                if args.use_pos_tags:
                    pos_tag_ids = np.zeros([n_sents, max_sent_len]) 
                if args.use_lemmas:
                    lemma_ids = np.zeros([n_sents, max_sent_len]) 
                if args.use_char_emb:
                    char_seq_map = {'<pad>': 0}
                    char_seqs = [[char_seq_map['<pad>']]]
                    char_seq_ids = []         
                    
                gold_tags = np.zeros([n_sents, max_sent_len]) 
                
                for i in range(n_sents):
                    char_ids = np.zeros([max_sent_len])
                    
                    # Next sentence                    
                    for j in range(sent_lens[i]):                    
                        token = batch[i][j]
                        
                        if args.use_l_m:
                            lm_emb[i, j] = token.embedding.numpy() # convert torch tensor to numpy array
                        if args.use_word_emb:
                            word_emb[i, j] = self.get_word_emb(token)  # get pretrained KeyedVector for each token
                        if args.use_words:
                            word = token.text
                            if word not in self.word_dict:
                                word = 'unk'
                            word_ids[i, j] = self.word_dict[word]
                        if args.use_pos_tags:
                            pos_tag_ids[i, j] = self.pos_tag_dict.get_idx_for_item(token.get_tag("upos").value)
                        if args.use_lemmas:
                            lemma_ids[i, j] = self.lemma_dict.get_idx_for_item(token.get_tag("lemma").value)                            
                        if args.use_char_emb:
                            if token.text not in char_seq_map:
                                char_seq = [self.char_dict.get_idx_for_item(c) for c in token.text]
                                char_seq_map[token.text] = len(char_seqs)
                                char_seqs.append(char_seq)                        
                            char_ids[j] = char_seq_map[token.text]     
                        gold_tags[i, j] = self.tag_dict.get_idx_for_item(token.get_tag(self.tag_type).value)      # tag index         
                                            
                    # Append sentence char_seq_ids
                    if args.use_char_emb:
                        char_seq_ids.append(char_ids)
                        
                # Run graph for batch
                feed_dict = {self.sentence_lens: sent_lens,
                             self.gold_tags: gold_tags, 
                             self.is_training: True}                              
                                        
                if args.use_l_m:      
                    feed_dict[self.lm_emb] = lm_emb  
                if args.use_word_emb:  
                    feed_dict[self.emb_words] = word_emb                  
                if args.use_words:
                    feed_dict[self.word_ids] = word_ids  
                if args.use_lemmas:
                    feed_dict[self.lemma_ids] = lemma_ids            
                if args.use_pos_tags:
                    feed_dict[self.pos_tag_ids] = pos_tag_ids 
                    
                # Pad char sequences                
                if args.use_char_emb:
                    char_seq_lens = [len(char_seq) for char_seq in char_seqs]
                    max_char_seq = max(char_seq_lens)
                    n = len(char_seqs)
                    batch_char_seqs = np.zeros([n, max_char_seq])
                    for i in range(n):
                        batch_char_seqs[i, 0:len(char_seqs[i])] = char_seqs[i]
                    
                    feed_dict[self.char_seqs] = batch_char_seqs
                    feed_dict[self.char_seq_ids] = char_seq_ids
                    feed_dict[self.char_seq_lens] = char_seq_lens                
                
                _, _, loss, predicted_tag_ids = self.session.run([self.training, self.summaries["train"], self.loss, self.predictions],
                                                                 feed_dict)
            
               
                # Add predicted tag to each token (annotate)    
                for i in range(n_sents):
                    for j in range(sent_lens[i]):
                        token = batch[i][j]
                        predicted_tag = self.tag_dict.get_item_for_index(predicted_tag_ids[i][j])
                        token.add_tag('predicted', predicted_tag)
    
                # Tally metrics        
                for sentence in batch:
                    gold_tags = [(tag.tag, str(tag)) for tag in sentence.get_spans(self.tag_type)]
                    predicted_tags = [(tag.tag, str(tag)) for tag in sentence.get_spans('predicted')]
       
                    for tag, pred in predicted_tags:
                        if (tag, pred) in gold_tags:
                            totals['tp'] += 1
                            totals_per_tag[tag]['tp'] += 1
                        else:
                            totals['fp'] +=1
                            totals_per_tag[tag]['fp'] += 1
                
                    for tag, gold in gold_tags:
                        if (tag, gold) not in predicted_tags:
                            totals['fn'] += 1
                            totals_per_tag[tag]['fn'] += 1  
                        else:
                            totals['tn'] +=1
                            totals_per_tag[tag]['tn'] += 1 # tn?
                 
                if not embeddings_in_memory:
                    self.clear_embeddings_in_batch(batch)                
                
            # Save model if checkpoints enabled
            if checkpoint:
                save_path = self.saver.save(self.session, "{}/checkpoint.ckpt".format(logdir))
                print("Checkpoint saved at ", save_path)
           
            # Save metrics for epoch
            self.metrics.log_metrics(dataset_name, totals, totals_per_tag, epoch, batch_n, self.scheduler.lr, self.scheduler.bad_epochs, dev_score)                
                   
            # Evaluate with dev data
            if train_with_dev:
                dev_data = corpus.dev                
                #dev_score = self.evaluate("dev", dev_data, dev_batch_size, epoch, embeddings_in_memory=embeddings_in_memory)
                score = self.evaluate(args, dev_data, "dev", epoch, embeddings_in_memory=embeddings_in_memory)
                 
                # Perform one step on lr scheduler
                self.scheduler.step(score)
            
            else:
                if metric == "accuracy":
                    score = self.metrics.accuracy
                else:
                    score =  self.metrics.f1
                    
                # Perform one step on lr scheduler
                self.scheduler.step(score)                    
            
                    
            #print("Epoch {} batch {}: train loss \t{}\t lr \t{}\t dev score \t{}\t bad epochs \t{}".format(epoch, batch_n, loss, self.lr, dev_score, self.scheduler.bad_epochs))        
            print("Epoch {} batch {}: train loss \t{}\t lr \t{}\t dev score \t{}\t bad epochs \t{}".format(epoch, batch_n, loss, self.scheduler.lr, dev_score, self.scheduler.bad_epochs))        
            
            # Save best model
            if score == self.scheduler.best:
                save_path = self.saver.save(self.session, "{}/best-model.ckpt".format(logdir))
                print("Best model saved at ", save_path) 
            
            

    def evaluate(self, args, dataset, dataset_name, epoch=None, test_mode=False, embeddings_in_memory=False, metric="accuracy"):                
    #def evaluate(self, dataset_name, dataset, eval_batch_size=32, epoch=None, test_mode=False, embeddings_in_memory=False, metric="accuracy"):
                
        print("Evaluating")
        
        self.session.run(self.reset_metrics)  # for batch statistics
        
        # Get batches
        batches = [dataset[x:x + args.eval_batch_size] for x in range(0, len(dataset), args.eval_batch_size)]
        
        # To store metrics
        totals_per_tag = defaultdict(lambda: defaultdict(int))
        totals = defaultdict(int)
        dev_loss = 0
        
        # Forward pass for each batch
        for batch_n, batch in enumerate(batches):
                          
            # Remove super long sentences which may cause memory issues      
            # Note mwe parseme data cannot be randomized
            if args.task != "mwe":
                
                # Sort batch and get lengths
                batch.sort(key=lambda i: len(i), reverse=True)
                max_sent_len = len(batch[0])                                    
                while len(batch) > 1 and max_sent_len > 200:
                    batch = batch[1:]
                    max_sent_len = len(batch[0])                    
            
      
            sent_lens = [len(s.tokens) for s in batch]
            max_sent_len = max(sent_lens)  
            n_sents = len(sent_lens)                      
            
            # Prepare embeddings, pad and embed sentences and tags
            
            # Embed sentences using Flair CharLM
            if args.use_l_m:
                self.lm.embed(batch)            
                lm_emb = np.zeros([n_sents, max_sent_len, self.lm_emb_dim])
           
            # For pretrained word embedding such as fastText    
            if args.use_word_emb:
                word_emb = np.zeros([n_sents, max_sent_len, self.word_emb_dim])
            
            # Prepare ids placeholders for trainable in-task embeddings 
            if args.use_words: 
                word_ids = np.zeros([n_sents, max_sent_len]) 
            if args.use_pos_tags:
                pos_tag_ids = np.zeros([n_sents, max_sent_len]) 
            if args.use_lemmas:
                lemma_ids = np.zeros([n_sents, max_sent_len]) 
            if args.use_char_emb:
                char_seq_map = {'<pad>': 0}
                char_seqs = [[char_seq_map['<pad>']]]
                char_seq_ids = []         
                
            gold_tags = np.zeros([n_sents, max_sent_len]) 
            
            for i in range(n_sents):
                char_ids = np.zeros([max_sent_len])
                # Next sentence                    
                for j in range(sent_lens[i]):                    
                    token = batch[i][j]
                    
                    if args.use_l_m:
                        lm_emb[i, j] = token.embedding.numpy() # convert torch tensor to numpy array
                    if args.use_word_emb:
                        word_emb[i, j] = self.get_word_emb(token)  # get pretrained KeyedVector for each token
                    if args.use_words:
                        #words[i, j] = self.word_dict.get_idx_for_item(token.get_tag("word").value) 
                        word = token.text
                        if word not in self.word_dict:
                            word = 'unk'
                        word_ids[i, j] = self.word_dict[word]       
                    if args.use_pos_tags:
                        pos_tag_ids[i, j] = self.pos_tag_dict.get_idx_for_item(token.get_tag("upos").value)
                    if args.use_lemmas:
                        lemma_ids[i, j] = self.lemma_dict.get_idx_for_item(token.get_tag("lemma").value)                            
                    if args.use_char_emb:
                        if token.text not in char_seq_map:
                            char_seq = [self.char_dict.get_idx_for_item(c) for c in token.text]
                            char_seq_map[token.text] = len(char_seqs)
                            char_seqs.append(char_seq)                        
                        char_ids[j] = char_seq_map[token.text]     
                    gold_tags[i, j] = self.tag_dict.get_idx_for_item(token.get_tag(self.tag_type).value)      # tag index         
                    
                # Append sentence char_seq_ids
                if args.use_char_emb:
                    char_seq_ids.append(char_ids)
                    
            # Run graph for batch
            feed_dict = {self.sentence_lens: sent_lens,
                         self.is_training: False}     
            
            if args.use_l_m:      
                feed_dict[self.lm_emb] = lm_emb  
            if args.use_word_emb:  
                feed_dict[self.emb_words] = word_emb                  
            if args.use_words:
                feed_dict[self.word_ids] = word_ids  
            if args.use_lemmas:
                feed_dict[self.lemma_ids] = lemma_ids            
            if args.use_pos_tags:
                feed_dict[self.pos_tag_ids] = pos_tag_ids 
       
            # Pad char sequences            
            if args.use_char_emb:
                #char_seqs.sort(key=lambda i: len(i), reverse=True)
                char_seq_lens = [len(char_seq) for char_seq in char_seqs]
                #max_char_seq = len(char_seqs[0])
                max_char_seq = max(char_seq_lens)
                n = len(char_seqs)
                batch_char_seqs = np.zeros([n, max_char_seq])
                for i in range(n):
                    batch_char_seqs[i, 0:len(char_seqs[i])] = char_seqs[i]
                
                feed_dict[self.char_seqs] = batch_char_seqs
                feed_dict[self.char_seq_ids] = char_seq_ids
                feed_dict[self.char_seq_lens] = char_seq_lens            
            
        
            # For dev data
            if not test_mode:
                feed_dict[self.gold_tags] = gold_tags  
                
                _, _, dev_loss, predicted_tag_ids =  self.session.run([self.summaries["dev"], 
                                                                       self.update_loss, 
                                                                       self.current_loss, 
                                                                       self.predictions],
                                                                      feed_dict)                
                print("dev loss ", dev_loss)
            
            # For test data
            else:
                _, predicted_tag_ids = self.session.run([self.summaries["test"], self.predictions],
                                                     feed_dict)                
                                       
            # Add predicted tag to each token (annotate)    
            for i in range(n_sents):
                for j in range(sent_lens[i]):
                    token = batch[i][j]
                    predicted_tag = self.tag_dict.get_item_for_index(predicted_tag_ids[i][j])
                    token.add_tag('predicted', predicted_tag)
           
            # Tally metrics        
            for sentence in batch:
                gold_tags = [(tag.tag, str(tag)) for tag in sentence.get_spans(self.tag_type)]
                predicted_tags = [(tag.tag, str(tag)) for tag in sentence.get_spans('predicted')]
                for tag, pred in predicted_tags:
                    if (tag, pred) in gold_tags:
                        totals['tp'] += 1
                        totals_per_tag[tag]['tp'] += 1
                    else:
                        totals['fp'] +=1
                        totals_per_tag[tag]['fp'] += 1
            
                for tag, gold in gold_tags:
                    if (tag, gold) not in predicted_tags:
                        totals['fn'] += 1
                        totals_per_tag[tag]['fn'] += 1  
                    else:
                        totals['tn'] +=1
                        totals_per_tag[tag]['tn'] += 1 # tn?
            
            if not embeddings_in_memory:
                self.clear_embeddings_in_batch(batch)             
        
        # Log dev metrics
        self.metrics.log_metrics(dataset_name, totals, totals_per_tag, epoch, batch_n, self.scheduler.lr, self.scheduler.bad_epochs)
        
                        
        # Write test results
        fh = "{}/tagger_test_".format(logdir) + str(dataset_name) + ".txt"        
        if test_mode:       
            # Write test results for pos or ner
            if args.task != "mwe":
                with open(fh, "w") as test_file:
                    for i in range(len(batches)):
                        for j in range(len(batches[i])): 
                            for k in range(len(batches[i][j])):
                                token = batches[i][j][k]
                                gold_tag = token.get_tag(self.tag_type).value
                                predicted_tag = token.get_tag('predicted').value
                                if predicted_tag != '':
                                    print("{} {} {}".format(token.text, gold_tag, predicted_tag), file=test_file)
                                
                                # This is if a long sentence was removed yielding no predicted tag here                                    
                                else:
                                    if self.tag_type == "ner":
                                        print("{} {} O".format(token.text, gold_tag), file=test_file)
                                    elif self.tag_type == "pos":
                                        print("{} {} N".format(token.text, gold_tag), file=test_file)
                            print("", file=test_file)
                #return
        
            # Write test results for mwe
            else:               
                with open(fh, "w") as test_file:
                    print("# global.columns = ID FORM NO_SPACE PARSEME:MWE", file=test_file)
                    for i in range(len(batches)):
                        for j in range(len(batches[i])): 
                            
                            # Convert tags back to cupti format
                            sent_spans = [(idx, span.tokens) for idx, span in enumerate(batches[i][j].get_spans("predicted"), 1)]
                            for idx, tokens in sent_spans:
                                first_t = tokens[0]
                                old_tag = first_t.get_tag("predicted").value
                                new_tag = str(idx) + ":" + old_tag[2:]
                                first_t.add_tag("predicted", new_tag)                                 
                                for token in tokens[1:]:
                                    token.add_tag("predicted", str(idx))                   
                            for k in range(len(batches[i][j])):
                                token = batches[i][j][k]
                                if predicted_tag != '':
                                    print("{}\t{}\t_\t{}".format(token.get_tag("idx").value,
                                                                 token.text, 
                                                                 token.get_tag("predicted").value),
                                          file=test_file)
                                
                                # This is if a long sentence was removed yielding no predicted tag here                                    
                                else:
                                    print("{}\t{}\t_\t_".format(token.get_tag("idx").value,
                                                                     token.text),
                                          file=test_file)
                            print("", file=test_file)
                #return        
        
        # Save and print metrics                  
        self.metrics.print_metrics()
        
        if metric == "accuracy":
            return self.metrics.accuracy
        
        else:
            return self.metrics.f1
    
    def get_word_emb(self, token):
        """Helper method to retrieve pretrained word emb"""
        
        if token.text in self.word_emb:
            word_embedding = self.word_emb[token.text]
        elif token.text.lower() in self.word_emb:
            word_embedding = self.word_emb[token.text.lower()]
        elif re.sub(r'\d', '#', token.text.lower()) in self.word_emb:
            word_embedding = self.word_emb[re.sub(r'\d', '#', token.text.lower())]
        elif re.sub(r'\d', '0', token.text.lower()) in self.word_emb:
            word_embedding = self.word_emb[re.sub(r'\d', '0', token.text.lower())]
        else:
            word_embedding = np.zeros(self.word_emb_dim, dtype='float')        
        return word_embedding
    
    def clear_embeddings_in_batch(self, batch):
        """Clearing stored emb will free up memory"""
        
        for sentence in batch:
            for token in sentence.tokens:
                token.clear_embeddings()      


    def word_dropout(self, x, dropout_rate):
        """
        Helper method to compute word dropout. Experimentation here, however,
        did not yield improved results
        """
        mask = tf.distributions.Bernoulli(1 - dropout_rate, 
                                          dtype=tf.float32).sample([tf.shape(x)[0], tf.shape(x)[1], 1])                                        
                
        return mask * x        

       
               
class Metrics:
    """Helper class to calculate and store metrics"""
    
    def __init__(self):
        
        self.accuracy = self.precision = self.recall = self.f1 = 0
    
    
    def log_metrics(self, dataset_name, totals, totals_per_tag, epoch, batch_n, lr=0, bad_epochs=0, dev_score=0):
        
        self.totals = totals
        self.totals_per_tag = totals_per_tag
        
        with open("{}/metrics-{}.txt".format(logdir, dataset_name), "a") as f:
           
            # Total metrics
            try: 
                self.accuracy = (float(totals['tp']) + totals['tn']) / (totals['tp'] + totals['fp'] + totals['tn'] + totals['fn'])
            except ZeroDivisionError: 
                self.accuracy = 0
            try: 
                self.precision = float(totals['tp']) / (totals['tp'] + totals['fp'])
            except ZeroDivisionError: 
                self.precision = 0             
            try: 
                self.recall = float(totals['tp']) / (totals['tp'] + totals['fn'])
            except ZeroDivisionError: 
                self.recall = 0            
            try:    
                self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)           
            except ZeroDivisionError: 
                self.f1 = 0            
            # Save
            f.write("\nEpoch {}\t Batch {}\t lr {}\tBad epochs {}\t Dev score {:.3f} \n".format(epoch, batch_n, lr, bad_epochs, dev_score ))                        
            f.write("tp {}\t fp {}\t tn {}\t fn {}\t acc {:.3f}\t prec {:.3f}\trec\t{:.3f}\tf1 {:.3f}\n".format(totals['tp'], totals['fp'], totals['tn'], totals['fn'], self.accuracy, self.precision, self.recall, self.f1))            
           
            # Metrics per tag
            for tag in totals_per_tag:
                try: 
                    accuracy = (float(totals_per_tag[tag]['tp']) + totals_per_tag[tag]['tn']) / (totals_per_tag[tag]['tp'] + totals_per_tag[tag]['fp'] + totals_per_tag[tag]['tn'] + totals_per_tag[tag]['fn'])
                except ZeroDivisionError: 
                    accuracy = 0
                try: 
                    precision = float(totals_per_tag[tag]['tp']) / (totals_per_tag[tag]['tp'] + totals_per_tag[tag]['fp'])
                except ZeroDivisionError: 
                    precision = 0
                try: 
                    recall = float(totals_per_tag[tag]['tp']) / (totals_per_tag[tag]['tp'] + totals_per_tag[tag]['fn'])
                except ZeroDivisionError: 
                    recall = 0
                try:    
                    f1 = 2 * precision * recall / (precision + recall)            
                except ZeroDivisionError: 
                    f1 = 0
                # Save
                f.write("{}\tacc {:.3f}\t prec {:.3f}\trec\t{:.3f}\tf1 {:.3f}\n".format(tag, accuracy, precision, recall, f1))            
     
    def print_metrics(self):
        print("acc {:.3f}\t prec {:.3f}\trec\t{:.3f}\tf1 {:.3f}\n".format(self.accuracy, self.precision, self.recall, self.f1))            


class ReduceLROnPlateau:
    """Reduce learning rate when a the loss has stopped improving.
        Models often benefit from reducing the learning rate by a factor
        of 2-10 once learning stagnates. This scheduler determines   
        whether to reduce the current learning rate if no change has
        been seen for a given number of epochs (the patience)
        Args:
            factor (float): Factor by which the learning rate will be
                reduced. new_lr = lr * factor. Default: 0.5.
            patience (int): Number of epochs with no improvement after
                which learning rate will be reduced. For example, if
                `patience = 5`, the first 5 epochs
                with no improvement will be ignored, and the lr will be
                reduced in the sixth epoch if the loss still hasn't improved then.
                Default: 10.
            verbose (bool): If ``True``, prints a message to stdout for
                each update. Default: ``False``.
            threshold (float): Threshold for measuring the new optimum,
                to only focus on significant changes. Default: 1e-4.
            threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
                dynamic_threshold = best * ( 1 + threshold ) in 'max'
                mode or best * ( 1 - threshold ) in `min` mode.
                In `abs` mode, dynamic_threshold = best + threshold in
                `max` mode or best - threshold in `min` mode. Default: 'rel'.
            eps (float): Minimal decay applied to lr. If the difference
                between new and old lr is smaller than eps, the update is
                ignored. Default: 1e-8.

            """
    def __init__(self, lr, factor=.5, patience=5, threshold=1e-4, threshold_mode="rel", eps=1e-8):
        self.lr = lr
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.eps = eps
        self.reset()

    def reset(self):
        self.bad_epochs = 0
        self.last_epoch = -1
        self.best = -inf

    def step(self, metric, epoch=None):
        epoch = self.last_epoch = self.last_epoch + 1
        cur = metric
        if self.is_better(cur, self.best):
            self.best = cur
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        if self.bad_epochs > self.patience:
            self.bad_epochs = 0
            return self.reduce_lr(epoch)
        else:
            return False

    def is_better(self, cur, best):
        if self.threshold_mode == "rel":
            eps = 1.0 - self.threshold
            return cur > best * eps
        elif self.threshold_mode == "abs":
            return cur > best - eps

    def reduce_lr(self, epoch):
        old_lr = self.lr
        new_lr = self.factor * old_lr
        if old_lr - new_lr > self.eps:
            self.lr = new_lr
            print("Epoch {:5d}: reducing learning rate from {} to {:.4e}.".format(epoch, old_lr, new_lr))                
            return True
        else:
            return False             


class CV:
    
    def __init__(self, args, corpus, lm=None, word_emb=None, k=5):
        
        print("Cross validating")
        
        #Train with CV
        data = corpus.train + corpus.dev + corpus.test  # concat data
        random.shuffle(data)
        n_sents = len(data)
        sents_per_fold = n_sents // k
        indices = [(i, i + sents_per_fold) for i in range(0, n_sents, sents_per_fold)
                   if i + sents_per_fold < n_sents] # indices to split
        
        print("No sents {}, sents per fold {}".format(n_sents, sents_per_fold))
        print("Indices", indices, len(indices))
        
        self.scores = []  
        iters = 0
        for i, j in indices:
            
            #Construct the tagger
            print("Constructing tagger, CV iter=", iter)
            model = SequenceTagger(args, corpus, lm=lm, word_emb=word_emb)            	
            train_data = []
            train_data.extend(data[0:i])
            train_data.extend(data[j:])
            test_data = data[i:j]        
            print("Training data size: ", len(train_data))
            print("Test data size: ", len(test_data))
            oov_rate(train_data, test_data)  # print OOV rate
            
            #Train without dev
            print("Training, CV iter={}".format(iter))
            model.train(args, train_data, "train_" + str(iters), checkpoint=True, embeddings_in_memory=False, train_with_dev=False, metric="f1")
            
            # Testing
            print("Testing, CV iter={}".format(iter))
            self.scores.append(model.evaluate(args, test_data, "test_" + str(iters), test_mode=True, embeddings_in_memory=False, metric="f1"))
            
            # Reset
            del model
            iters += 1
        
        print("CV scores\n", self.scores)
        
        @property
        def scores(self):
            return self.scores
        
        
def oov_rate(train_data, test_data):

    train_vocab = set([])
    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            train_vocab.add(train_data[i][j].text)

    print("Train vocab size: ", len(train_vocab))

    test_vocab = set([])
    for i in range(len(test_data)):
        for j in range(len(test_data[i])):
            test_vocab.add(test_data[i][j].text)                

    print("Test vocab size: ", len(test_vocab))

    diff = test_vocab.difference(train_vocab)

    print("Number of OOV words: ", len(diff))
    rate = len(diff) / len(test_vocab)
    print("OOV rate " , rate)     
    
    
if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import sys
    import re
    from flair.data_fetcher import NLPTaskDataFetcher
    #from flair.embeddings import CharLMEmbeddings, WordEmbeddings, StackedEmbeddings
    from flair.embeddings import StackedEmbeddings, FlairEmbeddings, WordEmbeddings
    from flair.data import Sentence, Dictionary
    from gensim.models import KeyedVectors    
    import numpy as np
    
    # Fix random seed
    np.random.seed(42)


    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size for dev.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_dim", default=256, type=int, help="RNN cell dimension.")    
    parser.add_argument("--char_emb_type", default="cnn", type=str, help="CNN or RNN char emb.")    
    parser.add_argument("--cnne_filters", default=200, type=int, help="# cnn filters")
    parser.add_argument("--cnne_max", default=3, type=int, help="Max filter size - 1")    
    parser.add_argument("--optimizer", default="SGD", type=str, help="Optimizer.")    
    parser.add_argument("--cle_dim", default=16, type=int, help="Character-level embedding dimension.")
    parser.add_argument("--epochs", default=150, type=int, help="Number of epochs.")
    parser.add_argument("--lr", default=0.1, type=float, help="Initial learning rate.") 
    parser.add_argument("--final_lr", default=.001, type=float, help="Final learning rate.")    
    parser.add_argument("--bn_input", default=0, type=int, help="Batch normalization.")
    parser.add_argument("--bn_char", default=0, type=int, help="Batch normalization.")
    parser.add_argument("--bn_sent", default=0, type=int, help="Batch normalization.")
    parser.add_argument("--bn_output", default=0, type=int, help="Batch normalization.")
    parser.add_argument("--dropout_input", default=0, type=float, help="Dropout rate.")
    parser.add_argument("--dropout_char", default=0, type=float, help="Dropout rate.")
    parser.add_argument("--dropout_sent", default=0, type=float, help="Dropout rate.")    
    parser.add_argument("--dropout_output", default=0, type=float, help="Dropout rate.")        
    parser.add_argument("--locked_dropout", default=0, type=float, help="Locked/Variational dropout rate.")
    parser.add_argument("--word_dropout", default=0, type=float, help="Word dropout rate.")
    parser.add_argument("--use_crf", default=1, type=int, help="Conditional random field decoder.")
    parser.add_argument("--use_pos_tags", default=0, type=int, help="In task PoS tag embeddings.")
    parser.add_argument("--use_words", default=0, type=int, help="In task word embeddings.")    
    parser.add_argument("--use_lemmas", default=0, type=int, help="In task lemma embeddings.")
    parser.add_argument("--use_char_emb", default=0, type=int, help="In task character level embeddings.")
    parser.add_argument("--use_word_emb", default=0, type=int, help="Pretrained word embeddings.")        
    parser.add_argument("--use_l_m", default=0, type=int, help="CharLM embeddings.")
    parser.add_argument("--downsize_lm", default=0, type=float, help="Downsize lm embeddings by this factor.")    
    parser.add_argument("--clip_gradient", default=.25, type=float, help="Norm for gradient clipping.")
    parser.add_argument("--annealing_factor", default=.5, type=float, help="LR will decay by this factor")    
    parser.add_argument("--patience", default=20, type=int, help="Patience for lr schedule.")    
    parser.add_argument("--task", default=None, type=str, help="Task.")
    parser.add_argument("--cv", default=None, type=int, help="Cross Validation.")
    
    args = parser.parse_args()
    filename = os.path.basename(__file__)
    
    # Create logdir name  
    logdir = "/home/lief/files/tagger/logs/{}-{}-{}".format(    
    #ogdir = "logs/{}-{}-{}".format(
        filename,
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    
    # Write file to directory    
    path_in = os.path.abspath(__file__)
    path_out = os.path.join(logdir, filename)    
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    with open(path_out, 'wt') as fout:
        with open(path_in, 'rt') as fin:
            for line in fin:
                fout.write(line)        

    #Get the corpus
    
    # if args.task == "pos":                                                                                                                   
    #     fh = "data/pos/macmorpho"
    #     cols = {0:"text", 1:"pos"} 
                                                                                                   
    # elif args.task == "ner":
    #     fh = "data/ner/harem" 
    #     cols = {0:"text", 1:"ner"}    
    
    # elif args.task == "ner-select":
    #     fh = "data/ner/harem/select" 
    #     cols = {0:"text", 1:"ner"}    

    # elif args.task == "mwe":
    #     fh = "data/pt/mwe" 
    #     cols = {0:"idx", 1:"text", 2:"lemma", 3:"upos", 4:"xpos", 5:"features", 6:"parent", 7:"deprel", 8:"deps", 9:"misc", 10:"mwe"}
     
    if args.task == "pos":
        fh = "/home/lief/tag/data/pos/macmorpho"
        cols = {0:"text", 1:"pos"}
  
    elif args.task == "ner":
        fh = "/home/lief/tag/data/ner/harem"
        cols = {0:"text", 1:"ner"}

    elif args.task == "ner-select":
        fh = "/home/lief/tag/data/ner/harem/select"
        cols = {0:"text", 1:"ner-select"}

    elif args.task == "mwe":
        #fh = "/home/lief/tag/data/mwe"
        fh = "/home/lief/data/data/mwe"
        cols = {0:"idx", 1:"text", 2:"lemma", 3:"upos", 4:"xpos", 5:"features", 6:"parent", 7:"deprel", 8:"deps", 9:"misc", 10:"mwe"}

     
    # Fetch corpus
    print("Getting corpus")
    corpus = NLPTaskDataFetcher.load_column_corpus(fh, 
                                                    cols, 
                                                    train_file="train.txt",
                                                    dev_file="dev.txt", 
                                                    test_file="test.txt")
    
    # Print some stats
    oov_rate(corpus.train, corpus.test)  # print OOV rate
    
    
    if args.use_word_emb:
        #word_emb_flair = WordEmbeddings("pt")                                                                               
        #word_emb = word_emb_flair.precomputed_word_embeddings # gensim emb                    
        word_emb = KeyedVectors.load("/home/lief/files/embeddings/cc.pt.300.kv")

    else:
        word_emb = None
        
    #Load Character Language Models (clms)                                                                              
    if args.use_l_m:
        #Stack lm embeddings
        fw_lm = FlairEmbeddings("/home/lief/lm/fw/best-lm.pt")
        bw_lm = FlairEmbeddings("/home/lief/lm/bw/best-lm.pt")
        lm =  StackedEmbeddings([fw_lm, bw_lm])


    else:
        lm = None
    
    
    
    # #embeddings = []
    # if args.use_word_emb:
    #     word_emb_flair = WordEmbeddings("pt")
    #     word_emb = word_emb_flair.precomputed_word_embeddings # gensim emb
    # else:
    #     word_emb = None
        
    # # Load Character Language Models (clms)
    # if args.use_l_m:
    #     # Stack lm embeddings
    #     lm = StackedEmbeddings([FlairEmbeddings("portuguese-forward"), FlairEmbeddings("portuguese-backward")])
    
    # else:
    #     lm = None
        
   
    #Train
    print("Beginning training") 
    if args.cv:
        CV(args, corpus, lm, word_emb, k=args.cv)
        
    else:
        
        # Construct the tagger
        print("Constructing tagger")
        tagger = SequenceTagger(args, corpus, lm=lm, word_emb=word_emb)                
        tagger.train(args, corpus.train, checkpoint=True, embeddings_in_memory=False, metric="accuracy")    
        
        # Test 
        tagger.evaluate(args, corpus.test, "test", test_mode=True, embeddings_in_memory=False, metric="accuracy")

     
