# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, 
# The HuggingFace Inc. team, and The XTREME Benchmark Authors.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#												http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions for NER/POS tagging tasks."""

from __future__ import absolute_import, division, print_function

import unicodedata as ud
import logging
import re
import string
import random
import numpy as np
import os
import copy
import math
from io import open
from transformers import XLMTokenizer
import time

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, langs=None, ipa_words=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
                                        specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.ipa_words = ipa_words
        self.langs = langs


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_ids, ipa_input_ids=None, input_emb_mask=None, ipa_emb_mask=None,mlm_ids=None, mlm_ipa_ids=None, mlm_label=None, mlm_ipa_label=None, ipa_features=None, xmlm_ids=None, xmlm_ipa_ids=None, xmlm_label=None, xmlm_ipa_label=None, lang_ids=None, normal_lang_ids=None, langs=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.langs = langs
        self.ipa_input_ids = ipa_input_ids

        self.input_emb_mask = input_emb_mask
        self.ipa_emb_mask = ipa_emb_mask
        self.mlm_ids = mlm_ids
        self.mlm_ipa_ids = mlm_ipa_ids
        self.mlm_label=mlm_label

        self.mlm_ipa_label=mlm_ipa_label


        self.xmlm_ids = xmlm_ids
        self.xmlm_ipa_ids = xmlm_ipa_ids
        self.xmlm_label=xmlm_label

        self.xmlm_ipa_label=xmlm_ipa_label

        self.ipa_features = ipa_features
        self.lang_ids = lang_ids

        self.normal_lang_ids = normal_lang_ids


def read_examples_from_file_pretrain(args,file_path, lang=None, lang2id=None):
    if not os.path.exists(file_path):
        logger.info("[Warming] file {} not exists".format(file_path))
        return []
    guid_index = 1
    examples = []
    subword_len_counter = 0
    if lang2id:
        lang_id = lang2id.get(lang, lang2id['en'])
    else:
        lang_id = 0
    logger.info("lang_id={}, lang={}, lang2id={}".format(lang_id, lang, lang2id))
    #print ("lang2id", lang2id)
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        langs = []
        ipa_words=[]
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if word:
                    examples.append(InputExample(guid="{}".format(guid_index),
                                                    words=words,
                                                    labels=labels,
                                                    langs=langs,
                                                    ipa_words=ipa_words))
                    guid_index += 1
                    words = []
                    labels = []
                    langs = []
                    ipa_words=[]
                    subword_len_counter = 0
                else:
                    print(f'guid_index', guid_index, words, langs, labels, subword_len_counter)
            else:
                splits = line.strip().split("\t") #[word, label (POS/NER), IPA, romanization, langs]

                assert len(splits) == 4                
                word = splits[0]

                words.append(splits[0])
                langs.append(lang_id)

                labels.append(splits[1])
                if (args.use_roman == 'true'):
                        ipa_words.append(splits[3])
                else:
                        ipa_words.append(splits[2])

        if words:
            examples.append(InputExample(guid="%d".format(guid_index),
                                            words=words,
                                            labels=labels,
                                            langs=langs,
                                            ipa_words=ipa_words))
        else:
            print ("Should not happen", words)
    return examples



def read_examples_from_file(args,file_path, lang, lang2id=None):
    if not os.path.exists(file_path):
        logger.info("[Warming] file {} not exists".format(file_path))
        return []
    guid_index = 1
    examples = []
    subword_len_counter = 0
    if lang2id:
        lang_id = lang2id.get(lang, lang2id['en'])
    else:
        lang_id = 0
    logger.info("lang_id={}, lang={}, lang2id={}".format(lang_id, lang, lang2id))
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        langs = []
        ipa_words=[]
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if word:
                    examples.append(InputExample(guid="{}-{}".format(lang, guid_index),
                                                    words=words,
                                                    labels=labels,
                                                    langs=langs,
                                                    ipa_words=ipa_words))
                    guid_index += 1
                    words = []
                    labels = []
                    langs = []
                    ipa_words=[]
                    subword_len_counter = 0
                else:
                    print(f'guid_index', guid_index, words, langs, labels, subword_len_counter)
            else:
                splits = line.strip().split("\t")
                word = splits[0]

                words.append(splits[0])
                langs.append(lang_id)
                labels.append(splits[1])
                if (args.use_roman == 'true'):
                        ipa_words.append(splits[3])
                else:
                        ipa_words.append(splits[2])

        if words:
            examples.append(InputExample(guid="%s-%d".format(lang, guid_index),
                                            words=words,
                                            labels=labels,
                                            langs=langs,
                                            ipa_words=ipa_words))
        else:
            print ("Should not happen", words)
    return examples

def compute_mlm_tokens(args, mlm_ipa_tokens, tokenizer, ipa_tokens, all_in_vocab,i, special_tokens_count):
    if (args.use_mlm_ratio == 'true'):
        #print ("ratio ipa")
        new_rand = random.random()
        if (new_rand <= 0.8):
            mlm_ipa_tokens.append(tokenizer.mask_token)
        elif (new_rand <= 0.9):
            mlm_ipa_tokens.append(ipa_tokens[i])
        else:

            rand_word = random.randint(0, len(all_in_vocab) -special_tokens_count)
            mlm_ipa_tokens.append(all_in_vocab[rand_word])
    else:
        mlm_ipa_tokens.append(tokenizer.mask_token)
    return mlm_ipa_tokens

#Return: selected_idx: idx that has been mapped to subtokens based on wwm_dict
def compute_selected_idx(args,label_ids_len,wwm_ipa_dict, special_tokens_count):				
    #Proposed Way
    rand = np.random.rand(len(wwm_ipa_dict))
    mask_arr = rand < args.mlm_ratio
    selected_w_idx = np.argwhere(mask_arr == True).reshape(-1,)

    if (len(selected_w_idx) <1): # none of them in mlm_ratio percentage, choose random 1 for very short sequence
        arange = np.arange(len(wwm_ipa_dict))
        np.random.shuffle(arange)
        selected_w_idx = arange[0]
        selected_w_idx = [selected_w_idx.tolist()]
                                                                
    selected_idx = []
    for j in selected_w_idx:				
        selected_idx.extend(wwm_ipa_dict[j])
    return selected_w_idx, selected_idx

#Used to generate MLM tokens and corresponding labels (either IPA or token side)
def generate_mlm_token_label(args, label_id_len, wwm_dict, tokens, tokenizer, all_in_vocab, special_tokens_count):
    selected_label_idx, selected_idx = compute_selected_idx(args,label_id_len,wwm_dict, special_tokens_count)
    
    mlm_label = []
    mlm_tokens=[]										
    for i in range (len(tokens)):
        if ( i in selected_idx):
            mlm_tokens = compute_mlm_tokens(args, mlm_tokens,tokenizer,tokens,all_in_vocab, i, special_tokens_count)
            mlm_label.append(tokenizer.convert_tokens_to_ids(tokens[i]))
        else:
            mlm_tokens.append(tokens[i])
            mlm_label.append(-100) #mask
                                                                    
    return mlm_tokens, mlm_label,selected_label_idx


def generate_xmlm(args, label_id_len, wwm_dict, wwm_ipa_dict, tokens, ipa_tokens, tokenizer, all_in_vocab, special_tokens_count):
    selected_label_idx, selected_idx = compute_selected_idx(args,label_id_len,wwm_dict, special_tokens_count)
    ipa_idx = []
    for j in selected_label_idx:
                                    ipa_idx.extend(wwm_ipa_dict[j])
    mlm_label = []
    mlm_tokens=[]										
    mlm_ipa_tokens=[]
    for i in range (len(tokens)):
        if ( i in selected_idx):
            mlm_tokens = compute_mlm_tokens(args, mlm_tokens,tokenizer,tokens,all_in_vocab, i, special_tokens_count)
            mlm_label.append(tokenizer.convert_tokens_to_ids(tokens[i]))
        else:
            mlm_tokens.append(tokens[i])
            mlm_label.append(-100) #mask
                                                                    
    return mlm_tokens, mlm_label

def tokenize_token_ipa(args, tokenizer, word, ipa_word, char_w_start, char_i_start):
    if isinstance(tokenizer, XLMTokenizer):
        word_tokens = tokenizer.tokenize(word, lang=lang)
    else:		
        word_tokens = tokenizer.tokenize(word)
        char_w_end = char_w_start + len(word_tokens)

        if (ipa_word.isdigit()):
            ipa_word_tokens = [ipa_word]
        else:
            if (args.model_type == 'bert'):
                ipa_word_tokens = tokenizer.tokenize(ipa_word)
                # Add sub-tokens symbols to punctuations (such as ˧)
                for idx in range (len(ipa_word_tokens)):
                    if (idx > 0):
                        if not (ipa_word_tokens[idx].startswith('##')):
                            ipa_word_tokens[idx] = '##' + ipa_word_tokens[idx]
            else:
                ipa_word_tokens = tokenizer.tokenize(ipa_word)
        char_i_end = char_i_start + len(ipa_word_tokens)

    if len(word) != 0 and len(word_tokens) == 0:
        word_tokens = [tokenizer.unk_token]


    if len(word) == 0 and len(word_tokens) == 0:
        word_tokens = [tokenizer.unk_token]

    if len(ipa_word) != 0 and len(ipa_word_tokens) == 0:
        ipa_word_tokens = [tokenizer.unk_token]

    if len(ipa_word) == 0 and len(ipa_word_tokens) == 0:
        ipa_word_tokens = [tokenizer.unk_token]
    
    return word_tokens, ipa_word_tokens, char_w_end, char_i_end

def convert_examples_to_features(args, mode, examples,
        label_list,
        max_seq_length,
        tokenizer,
        word_dict,
        ipa_dict,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-1,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        lang='en',
        langid=None):
    """ Loads a data file into a list of `InputBatch`s
                                    `cls_token_at_end` define the location of the CLS token:
                                                                    - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                                                                    - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
                                    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """


    special_tokens_count = 3 if sep_token_extra else 2
    
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    word_count=[]
    w_len=[]
    start_time = time.time()

    word_len_stat=[]
    tok_len_stat, ipa_tok_len_stat=[],[]
    cs_accumulator = []
    acc_cs, acc_word = 0,0
    cutoff_limit = max_seq_length - special_tokens_count

    if (args.add_mlm =='true' or args.add_xmlm == 'true'):
        all_in_vocab = list(set(tokenizer.get_vocab().keys()))

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        start_ex_time = time.time()
        tokens, ipa_tokens = [], []
        ipa_label_ids = []
        x_tokens, x_ipa_tokens = [], []
        label_ids = []
        full_seg_vec=[]
        ipa_example_label_ids=[]
        token_emb_mask, ipa_emb_mask, x_emb_mask =[], [], []
        ori_text =	example.words + example.ipa_words
        w_count = 0
        num_label=0
        char_w_start=char_w_end=0
        char_i_start=char_i_end=0

        char_wx_start=char_wx_end=0
        char_ix_start=char_ix_end=0


        if (args.use_new_pooling == 'true'):
            pool_id_cls = 0
        else:
            pool_id_cls=-1

        c_count=0
        w_len.append(len(example.words))
        wwm_dict, wwm_ipa_dict ={} , {}
        wwm_x_dict, wwm_x_ipa_dict = {}, {}
        cum_ex_time, cum_mlm_time=0.0,0.0
        idx_info_dict={}

        lang_ids=[] #either 0 or 1 (0 =src, 1 = tgt)

        word_len_stat.append(len(example.words))
        tok_acc, ipa_acc = 0,0
        x_tok_acc, x_ipa_acc = 0,0
        

        iter_words, iter_labels, iter_ipa = example.words, example.labels, example.ipa_words
        # Normal case of bilingual
        if (type (lang) == str):
            d_id = langid[lang]
            llangs=[d_id]* len(iter_ipa)
        else: #Next: multilingual pretraining
            pass

        seg_iter = []
        for word, label, ipa_word,default_l_id in zip(iter_words, iter_labels, iter_ipa, llangs):		
            word = word.replace('_', ' ')
            ipa_word = ipa_word.replace('_', ' ')
            start_1 = time.time() 

            acc_word += 1
            if (args.add_xmlm == 'true'):
                if (random.random() <= args.cs_ratio): # translation_dict={'src_word':[(tgt_w, tgt_ipa), (tgt_w2, tgt_ipa2),...]}
                    if (word in word_dict):
                        all_translations = word_dict[word]
                        if (len(all_translations) == 1): # only 1 translation
                            x_word = all_translations[0][0]
                            x_ipa_word = all_translations[0][1]
                                                    
                        else:
                            all_translations = np.array(all_translations)
                            np.random.shuffle(all_translations)
                            x_word = all_translations[0][0]
                            x_ipa_word = all_translations[0][1]

                        l_id = [langid[args.predict_langs]] * len(word.split(' '))

                        c_count += 1
                        acc_cs += 1
                    else:
                        #lang_ids += [l_id]			
                        x_word = word
                        x_ipa_word = ipa_word
                        l_id = [default_l_id]
                                                
                else:
                    #lang_ids += [l_id]			
                    x_word = word
                    x_ipa_word = ipa_word
                    l_id = [default_l_id]

            else:
                #lang_ids.append(l_id)
                x_word = word
                x_ipa_word = ipa_word
                l_id = [default_l_id]
            
            word_tokens, ipa_word_tokens, char_w_end, char_i_end = tokenize_token_ipa(args, tokenizer, word, ipa_word, char_w_start, char_i_start)
            x_word_tokens, x_ipa_word_tokens, char_wx_end, char_ix_end = tokenize_token_ipa(args, tokenizer, x_word, x_ipa_word, char_wx_start, char_ix_start)


            #TODO: Linguistic phonemic Features integration
            seg_vec = np.zeros((1,24))
            #seg_vec.append(np.zeros((24)))
            #seg_vec = compute_phonetic_features(ipa_word)
            full_seg_vec.append(seg_vec)
                                            
            tok_acc += len(word_tokens)
            ipa_acc += len(ipa_word_tokens)

            x_tok_acc += len(x_word_tokens)
            x_ipa_acc += len(x_ipa_word_tokens)

            normal_cond = ((tok_acc <= cutoff_limit) and (ipa_acc <= cutoff_limit)) 
            #normal_cond = (tok_acc <= (max_seq_length - special_tokens_count)) and (ipa_acc <= (max_seq_length - special_tokens_count))

            # Conduct cut-off to avoid label-token misalignment (in pooling)
            if (args.add_xmlm == 'true'):
                normal_cond = normal_cond and ((x_tok_acc <= cutoff_limit) and (x_ipa_acc <= cutoff_limit)) 
            if (normal_cond):
                seg_iter.append(word)
                tokens.extend(word_tokens)
                ipa_tokens.extend(ipa_word_tokens)
                x_tokens.extend(x_word_tokens)
                x_ipa_tokens.extend(x_ipa_word_tokens)
                if (isinstance(l_id, int)):
                    lang_ids.append(l_id)
                else:
                    lang_ids += l_id				

                #------ App2: Separate Masking for IPA and SUBWORD, aligning by WORD/ CHARACTER LEVEL ------#
                # Alignment IPA & Word tokenizers
                sub_token_range = np.arange(char_w_start, char_w_end).tolist()
                sub_ipa_range = np.arange(char_i_start, char_i_end).tolist()

                
                sub_x_range = np.arange(char_wx_start, char_wx_end).tolist()
                sub_xi_range = np.arange(char_ix_start, char_ix_end).tolist()

                wwm_dict[w_count] =sub_token_range
                wwm_ipa_dict[w_count] =sub_ipa_range

                wwm_x_dict[w_count] =sub_x_range
                wwm_x_ipa_dict[w_count] =sub_xi_range

                char_w_start = char_w_end
                char_i_start = char_i_end
                char_wx_start = char_wx_end
                char_ix_start = char_ix_end

                if (args.use_new_pooling == 'true'):
                    token_emb_mask.extend([len(token_emb_mask) + 1]* len(word_tokens)) # new pooling way
                    ipa_emb_mask.extend([len(ipa_emb_mask) + 1]* len(ipa_word_tokens)) # new pooling way
                    x_emb_mask.extend([len(x_emb_mask) + 1]* len(x_word_tokens))
                else:
                # My original way
                    token_emb_mask.extend([w_count]* len(word_tokens))
                    ipa_emb_mask.extend([w_count]* len(ipa_word_tokens))
                    x_emb_mask.extend([w_count]* len(x_word_tokens))
                num_label+=1
                w_count += 1

                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label]])
                ipa_example_label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(ipa_tokens) - 1))
                #ipa_label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(ipa_word_tokens) - 1))
                ipa_label_ids.extend([pad_token_label_id]* len(ipa_word_tokens)) #ipa label should not contribute to the loss and predictions
        assert len(ipa_emb_mask) == len(ipa_tokens)
        assert len(token_emb_mask) == len(tokens)
        assert len(x_emb_mask) == len(x_tokens)
        ratio_cs = c_count / w_count
        cs_accumulator.append(ratio_cs)
        word_count.append(c_count)
        tok_len_stat.append(len(tokens))
        ipa_tok_len_stat.append(len(ipa_tokens))				

        final_wwm_dict = wwm_dict
        final_wwm_ipa_dict = wwm_ipa_dict
        final_wwm_x_dict = wwm_x_dict
        final_wwm_x_ipa_dict = wwm_x_ipa_dict


        #--- END USING SEG-MLM -----#

        unique = np.unique(np.array(token_emb_mask))
        ipa_unique = np.unique(np.array(ipa_emb_mask))
        assert len(label_ids) == len(unique)
        assert len(label_ids) == len(ipa_unique)

        start_gen= time.time()

        rand_score = random.random()
        if (args.add_mlm== 'true'):
            #-- Mask IPA		

            mlm_tokens, mlm_label, selected_label_idx = generate_mlm_token_label(args, len(label_ids), final_wwm_dict, tokens, tokenizer,all_in_vocab, special_tokens_count)
            mlm_ipa_tokens =copy.deepcopy(ipa_tokens)				
            mlm_ipa_label = [-100] * len(mlm_ipa_tokens)

            assert len(mlm_ipa_tokens) == len(ipa_tokens)
            assert len(mlm_tokens) == len(tokens)
            assert (len(mlm_label) == len(mlm_tokens))
            assert (len(mlm_ipa_label) == len(mlm_ipa_tokens))

            mlm_time = time.time() - start_gen
        else:
            #print ("--Pass masking--")
            mlm_ipa_tokens = [tokenizer.unk_token]* len(ipa_tokens)
            mlm_tokens = [tokenizer.unk_token] * len(tokens)
            mlm_label = [-1] * len(tokens)
            mlm_ipa_label = [-1] * len(ipa_tokens)

            mlm_time=0.0

        #---- XMLM Input Preparation -----#
        if (args.add_xmlm == 'true'):
            # If MLM is not done, generate new MLM locations
            if (args.add_mlm == 'false'):
                #mlm_tokens, mlm_label, selected_label_idx = generate_mlm_token_label(args, len(label_ids), final_wwm_dict, tokens, tokenizer,all_in_vocab, special_tokens_count)
                xmlm_tokens, xmlm_label,_ = generate_mlm_token_label(args, len(label_ids), final_wwm_x_dict, x_tokens, tokenizer, all_in_vocab, special_tokens_count)
                xmlm_ipa_tokens = copy.deepcopy(ipa_tokens)
                xmlm_ipa_label = [-100] * len(xmlm_ipa_tokens)
            else:
                #print ("Align")
                x_idx = []
                for j in selected_label_idx:
                    x_idx.extend(final_wwm_x_dict[j])
                xmlm_label = []
                xmlm_tokens=[]									

                for i in range (len(x_tokens)):
                    if ( i in x_idx):
                        xmlm_tokens = compute_mlm_tokens(args, xmlm_tokens,tokenizer,x_tokens,all_in_vocab, i, special_tokens_count)
                        xmlm_label.append(tokenizer.convert_tokens_to_ids(x_tokens[i]))
                    else:
                        xmlm_tokens.append(x_tokens[i])
                        xmlm_label.append(-100) #mask
                xmlm_ipa_tokens = x_ipa_tokens
                xmlm_ipa_label = [-100]* len(xmlm_ipa_tokens)#already defined

            # Sanity check XMLM
            assert len(xmlm_ipa_tokens) == len(x_ipa_tokens)
            assert len(xmlm_tokens) == len(x_tokens)
            assert (len(xmlm_label) == len(xmlm_tokens))
            assert (len(xmlm_ipa_label) == len(xmlm_ipa_tokens))
                                                                        
        else:
            xmlm_ipa_tokens = [tokenizer.unk_token]* len(ipa_tokens)
            xmlm_tokens = [tokenizer.unk_token] * len(tokens)
            xmlm_label = [-1] * len(tokens)
            xmlm_ipa_label = [-1] * len(ipa_tokens)

 
        #--- PADDING CONVENTIONS				
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:  [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:    0	0			0									0							0												0																0								0								1  1		1  1			1				1
        # (b) For single sequences:
        #  tokens:				[CLS] the dog is hairy . [SEP]
        #  type_ids:			0				0				0				0			0												0								0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.

        tokens += [sep_token]
        ipa_tokens += [sep_token]
        full_seg_vec.append(np.zeros((1,24)))
        label_ids += [pad_token_label_id]

        lang_ids +=l_id
        #lang_ids += [0]
        ipa_label_ids += [pad_token_label_id]

        ipa_example_label_ids += [pad_token_label_id]
       

        token_emb_mask += [pool_id_cls]
        ipa_emb_mask += [pool_id_cls]

        mlm_ipa_tokens += [sep_token]
        mlm_tokens += [sep_token]
        mlm_label += [pad_token_label_id]
        mlm_ipa_label += [pad_token_label_id]


        xmlm_ipa_tokens += [sep_token]
        xmlm_tokens += [sep_token]
        xmlm_label += [pad_token_label_id]
        xmlm_ipa_label += [pad_token_label_id]

        # Fix when applying XLM-R
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            ipa_tokens += [sep_token]
            label_ids += [pad_token_label_id]

            ipa_label_ids += [pad_token_label_id]

            token_emb_mask += [pool_id_cls]
            ipa_emb_mask += [pool_id_cls]
            mlm_ipa_tokens += [sep_token]
            mlm_tokens += [sep_token]
            mlm_label += [pad_token_label_id]
            mlm_ipa_label += [pad_token_label_id]

            #lang_ids += [0]
            lang_ids +=l_id

        segment_ids = [sequence_a_segment_id] * len(label_ids)

        if cls_token_at_end:
            tokens += [cls_token]
            ipa_tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            ipa_tokens = [cls_token] + ipa_tokens

            full_seg_vec.insert(0,np.zeros((1,24)))
            label_ids = [pad_token_label_id] + label_ids


            lang_ids = l_id + lang_ids

            ipa_label_ids = [pad_token_label_id] + ipa_label_ids

            ipa_example_label_ids = [pad_token_label_id] + ipa_example_label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

            # Form MLM
            mlm_ipa_tokens = [cls_token] + mlm_ipa_tokens
            mlm_tokens = [cls_token] + mlm_tokens
            mlm_label = [pad_token_label_id] + mlm_label
            mlm_ipa_label = [pad_token_label_id] + mlm_ipa_label


            #For XMLM
            xmlm_ipa_tokens = [cls_token] + xmlm_ipa_tokens
            xmlm_tokens = [cls_token] + xmlm_tokens
            xmlm_label = [pad_token_label_id] + xmlm_label
            xmlm_ipa_label = [pad_token_label_id] + xmlm_ipa_label


            # For masking to convert subword -> word
            token_emb_mask = [pool_id_cls] + token_emb_mask
            ipa_emb_mask = [pool_id_cls] + ipa_emb_mask


        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        ipa_input_ids = tokenizer.convert_tokens_to_ids(ipa_tokens)

        mlm_ids = tokenizer.convert_tokens_to_ids(mlm_tokens)
        mlm_ipa_ids = tokenizer.convert_tokens_to_ids(mlm_ipa_tokens)

        xmlm_ids = tokenizer.convert_tokens_to_ids(xmlm_tokens)
        xmlm_ipa_ids = tokenizer.convert_tokens_to_ids(xmlm_ipa_tokens)

        #--- undo when switch back to first token + mask subtokens -----#
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        #input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # One per word
        input_mask = [1 if mask_padding_with_zero else 0] * len(label_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:

            input_ids += ([pad_token] * padding_length)

            #New padding length (for each label)
            new_pad = max_seq_length - len(label_ids)				
            label_ids += ([pad_token_label_id] * (new_pad))

            lang_ids += (l_id* (new_pad))

            input_mask += ([0 if mask_padding_with_zero else 1] * (new_pad))
            segment_ids += ([pad_token_segment_id] * (new_pad))

            ipa_pad = max_seq_length - len(ipa_input_ids)
            ipa_input_ids += ([pad_token] * ipa_pad)
            ipa_label_ids += ([pad_token_label_id] * ipa_pad)
            ipa_example_label_ids += ([pad_token_label_id] * ipa_pad)

            full_seg_vec.append(np.zeros((max_seq_length - len(full_seg_vec),24)))

            
            token_emb_mask += ([pool_id_cls] * padding_length)
            ipa_emb_mask += ([pool_id_cls] * ipa_pad)

            #for MLM objectives
            mlm_ipa_ids += ([pad_token] *ipa_pad)
            mlm_ids += ([pad_token] * padding_length)

            x_pad = max_seq_length - len (xmlm_ids)

            x_ipa_pad = max_seq_length - len (xmlm_ipa_ids)
            xmlm_ids += ([pad_token] * x_pad)
            xmlm_ipa_ids += ([pad_token] *x_ipa_pad)


            xmlm_ipa_label += ([pad_token_label_id] *x_ipa_pad)
            xmlm_label += ([pad_token_label_id] *x_pad)
            
            
            if (args.add_mlm == 'true'):

                mlm_ipa_label += ([pad_token_label_id] *ipa_pad)
                mlm_label += ([pad_token_label_id] *padding_length)
            else:
                mlm_label += ([pad_token_label_id] *padding_length)
                mlm_ipa_label += ([pad_token_label_id] *ipa_pad)

        if example.langs and len(example.langs) > 0:
            langs = [example.langs[0]] * max_seq_length
        else:
            print('example.langs', example.langs, example.words, len(example.langs))
            print('ex_index', ex_index, len(examples))
            langs = None
        full_seg_vec = np.concatenate((full_seg_vec),0)
        full_seg_vec = np.expand_dims(full_seg_vec,0)
        normal_lang_ids = l_id * max_seq_length
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(langs) == max_seq_length
        assert len(lang_ids) == max_seq_length

        
        assert full_seg_vec.shape[1] == max_seq_length #[1,128,24]
        assert len(ipa_input_ids) == max_seq_length

        if (args.add_mlm == 'true'):
            # Check MLM Input
            assert len(mlm_ipa_ids) == max_seq_length
            assert len(mlm_ids) == max_seq_length
            assert len(mlm_label) == max_seq_length

        if (args.add_xmlm == 'true'): 
            assert len(xmlm_ipa_ids) == max_seq_length 
            assert len(xmlm_ids) == max_seq_length
            assert len(xmlm_label) == max_seq_length
        
        assert len(ipa_emb_mask) == max_seq_length
        assert len(ipa_emb_mask) == max_seq_length == len(token_emb_mask)

        # --- Logging ----#
        if (ex_index < 1 and ((mode == 'train') or(mode=='test' and (lang =='ko' or lang == 'vi')))):
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("ipa tokens: %s", " ".join([str(x) for x in ipa_tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("token_emb_mask: %s", " ".join([str(x) for x in token_emb_mask]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("ipa_ids: %s", " ".join([str(x) for x in ipa_input_ids]))
            logger.info("ipa_emb_mask: %s", " ".join([str(x) for x in ipa_emb_mask]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            logger.info("normal_lang_ids: %s", " ".join([str(x) for x in normal_lang_ids]))

            if (args.add_mlm == 'true'):
                logger.info("*** MLM Info ***")

                logger.info("mlm_tokens: %s", " ".join([str(x) for x in mlm_tokens]))
                logger.info("mlm_ipa_tokens: %s", " ".join([str(x) for x in mlm_ipa_tokens]))
                logger.info("mlm_ids: %s", " ".join([str(x) for x in mlm_ids]))
                logger.info("mlm_ipa_ids: %s", " ".join([str(x) for x in mlm_ipa_ids]))
                logger.info("mlm_labels: %s", " ".join([str(x) for x in mlm_label]))
                logger.info("mlm_ipa_labels: %s", " ".join([str(x) for x in mlm_ipa_label]))
                logger.info("normal_lang_ids: %s", " ".join([str(x) for x in normal_lang_ids]))

            if (args.add_xmlm == 'true'):

                logger.info("*** X-MLM Info ***")

                logger.info("(non-mask) xmlm_tokens: %s", " ".join([str(x) for x in x_tokens]))
                logger.info("xmlm_tokens: %s", " ".join([str(x) for x in xmlm_tokens]))
                logger.info("xmlm_ipa_tokens: %s", " ".join([str(x) for x in xmlm_ipa_tokens]))
                logger.info("xmlm_ids: %s", " ".join([str(x) for x in xmlm_ids]))
                logger.info("xmlm_ipa_ids: %s", " ".join([str(x) for x in xmlm_ipa_ids]))
                logger.info("xmlm_labels: %s", " ".join([str(x) for x in xmlm_label]))
                logger.info("xmlm_ipa_labels: %s", " ".join([str(x) for x in xmlm_ipa_label]))
                logger.info("lang_ids: %s", " ".join([str(x) for x in lang_ids]))


        features.append(InputFeatures(input_ids=input_ids,
                                        input_mask=input_mask,
                                        segment_ids=segment_ids,
                                        label_ids=label_ids,
                                        langs=langs,
                                        ipa_input_ids=ipa_input_ids,
                                        input_emb_mask = token_emb_mask,
                                        ipa_emb_mask = ipa_emb_mask,
                                        mlm_ids = mlm_ids,
                                        mlm_ipa_ids = mlm_ipa_ids,
                                        mlm_label = mlm_label,
                                        mlm_ipa_label = mlm_ipa_label,
                                        xmlm_ids = xmlm_ids,
                                        xmlm_ipa_ids = xmlm_ipa_ids,
                                        xmlm_label = xmlm_label,
                                        xmlm_ipa_label = xmlm_ipa_label,
                                        ipa_features = full_seg_vec,
                                        lang_ids = lang_ids,
                                        normal_lang_ids = normal_lang_ids
))
        
        one_ex_time = time.time() - start_ex_time

        cum_ex_time += one_ex_time
        cum_mlm_time += mlm_time
    word_count = np.array(word_count)

    cs_accumulator = np.array(cs_accumulator)

    tok_len_stat, ipa_tok_len_stat, np_word_len_stat = np.array(tok_len_stat), np.array(ipa_tok_len_stat), np.array(word_len_stat)
    print ("Mean token length", np.mean(tok_len_stat))

    print ("Word len Tok-WORD IPA-TOK",np.mean(np_word_len_stat), np.mean(tok_len_stat), np.mean(ipa_tok_len_stat)) 
    print ("Finish reading examples", time.time()-start_time)
    return features


def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels
