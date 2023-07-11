import torch
import torch.nn as nn
import os
import numpy as np

lang_loc={'en':0, 'ja':3,'ko':6, 'vi':9,'zh':12}
def load_dictionary_pretrain(args): 
        dict_path ='../' + 'ipa_unified_dict.txt'
        #STRUCTURE: dict ={'word':['ipa1', 'ipa2',etc.], 'word2':[]}
        #MULTILINGUAL STRUCTURE: {'word':[[w_lang1, w_lang2, w_lang3,w_lang4], [ipa_lang1, ipa_lang2, ipa_lang3], word2:[]]


        word_dict, inv_word_dict={},{} #{word_dict ={ortho_word:{}}}
        ipa_dict, inv_ipa_dict ={}, {}
        src, tgt = args.train_langs, args.predict_langs
        src_loc, tgt_loc = lang_loc[src], lang_loc[tgt]
        all_langs=['zh','vi','ja','ko']
        all_langs = sorted(all_langs)
        word_tuple=[]

        lang_dict={'ja':{}, 'ko':{}, 'vi':{}, 'zh':{}}
        for line in open(dict_path, 'r'):
                elements = line.strip().split('\t') #src_ori,src_ipa, tgt_ori, tgt_ipa
 
                # For multilingual case
                for llang in all_langs: 
                    l_word = elements[lang_loc[llang]]
                    rem_lang =  all_langs.remove(llang)
                    rem_lang = sorted(rem_lang)
                    dict_info=[]
                    for rem in rem_lang:
                        w, ipa = elements[lang_loc[rem]], elements[lang_loc[rem]]
                        dict_info.append((w,ipa))

                    if (l_word in lang_dict[llang]):

                        lang_dict[llang][l_word]= dict_info
                    else:
                        lang_dict[llang][l_word].append(dict_info)

                src_word_corr, tgt_word_corr = elements[src_loc],elements[tgt_loc]

                if (args.use_roman == 'false'):

                    #print ("IPA dict")
                    src_ipa_corr, tgt_ipa_corr = elements[src_loc + 2], elements[tgt_loc + 2] #ipa  = ori + 2
                else:
                    #print ("Roman dict")
                    src_ipa_corr, tgt_ipa_corr = elements[src_loc + 1], elements[tgt_loc + 1] # roman = ori + 1
 
  
                if (src_word_corr not in word_dict):
                        word_dict[src_word_corr] = [tgt_word_corr]
                else:
                        word_dict[src_word_corr].append(tgt_word_corr)
 
                if (tgt_word_corr not in inv_word_dict):
                        inv_word_dict[src_word_corr] = [src_word_corr]
                else:
                        inv_word_dict[tgt_word_corr].append(src_word_corr)
  
                if (src_word_corr not in ipa_dict):
                        ipa_dict[src_word_corr] = [tuple((tgt_word_corr, tgt_ipa_corr))]
                else:
                        ipa_dict[src_word_corr].append(tuple((tgt_word_corr, tgt_ipa_corr)))
 
  
                if (tgt_word_corr not in inv_ipa_dict):
                        inv_ipa_dict[tgt_word_corr] = [tuple((src_word_corr, src_ipa_corr))]
                else:
                        inv_ipa_dict[tgt_word_corr].append(tuple((src_word_corr, src_ipa_corr)))
        
        lang_dict={'ja':ja_dict, 'ko':ko_dict, 'vi':vi_dict, 'zh':zh_dict}
 
        return word_dict, inv_word_dict,ipa_dict, inv_ipa_dict

def load_updated_dictionary(args):
        src, tgt = args.train_langs, args.predict_langs
        src_loc, tgt_loc = lang_loc[src], lang_loc[tgt]
        dict_path = '/hdd/multi_project/' + args.train_langs + '-' + args.predict_langs + '-dict.txt'
        word_dict, inv_word_dict={},{}
        ipa_dict, inv_ipa_dict ={}, {}
        for line in open(dict_path, 'r'):
                elements = line.strip().split('\t') #src_ori,src_ipa, tgt_ori, tgt_ipa
                src_word_corr, tgt_word_corr = elements[src_loc],elements[tgt_loc]

                if (args.use_roman == 'false'):
                    src_ipa_corr, tgt_ipa_corr = elements[src_loc + 2], elements[tgt_loc + 2] #ipa  = ori + 2
                else:
                    src_ipa_corr, tgt_ipa_corr = elements[src_loc + 1], elements[tgt_loc + 1] # roman = ori + 1
 
 
                if (src_word_corr not in word_dict):
                        word_dict[src_word_corr] = [tgt_word_corr]
                else:
                        word_dict[src_word_corr].append(tgt_word_corr)
 
                if (tgt_word_corr not in inv_word_dict):
                        inv_word_dict[src_word_corr] = [src_word_corr]
                else:
                        inv_word_dict[tgt_word_corr].append(src_word_corr)
 
 
                if (src_word_corr not in ipa_dict):
                        ipa_dict[src_word_corr] = [tuple((tgt_word_corr, tgt_ipa_corr))]
                else:
                        ipa_dict[src_word_corr].append(tuple((tgt_word_corr, tgt_ipa_corr)))
  
                if (tgt_word_corr not in inv_ipa_dict):
                        inv_ipa_dict[tgt_word_corr] = [tuple((src_word_corr, src_ipa_corr))]
                else:
                        inv_ipa_dict[tgt_word_corr].append(tuple((src_word_corr, src_ipa_corr)))
 
        return word_dict, inv_word_dict,ipa_dict, inv_ipa_dict
 


def compute_align(self, outputs, labels):                                   
    def prepare_label(input_align, mask, config):
        labels = torch.arange(input_align.shape[1])     #[max_len (128)]
        labels = labels.to(input_align.get_device())
        labels = labels.unsqueeze(0) #[1,128]
        labels = torch.repeat_interleave(labels,input_align.shape[0], dim=0) #[8,128]]
        #If use no_extra
        labels = labels * mask

        labels = labels.long()
        input_ = torch.count_nonzero(input_align.sum(dim=-1),dim=-1)
        assert torch.allclose(torch.amax(labels,1) + 1, torch.count_nonzero(input_align.sum(dim=-1),dim=-1))
        labels[labels == 0.0] = -100
                     
        if (self.config.cls_alignment == 'true'):
                #If consider full seq
                labels[:,0]=0.0 #[CLS] a b c [SEP] are considered
                assert torch.allclose(torch.count_nonzero(input_align.sum(dim=-1),dim=-1), torch.amax(labels, 1) + 1) #Assumption: max(ipa_emb_mask) + 1 (total number of tokens) + separate_tokens (2 for mbert) = ipa_
                 

        return labels
                     
    input_align, ipa_align, mask_align = outputs.input_embed, outputs.ipa_embed, outputs.mask_embed
    align_labels = prepare_label(input_align, mask_align, self.config)
                     
    #--- IPA-Token Alignment Loss
    #if (ipa_ids is not None and self.config.add_alignment == 'true'):      
    if (self.config.add_alignment == 'true'):
            #pass    
            assert self.config.use_ipa == 'true'
            input_, ipa_ = torch.count_nonzero(input_align.sum(dim=-1), dim=-1), torch.count_nonzero(ipa_align.sum(dim=-1),dim=-1) #[num_act_token + num_sep_tokens]
            assert torch.allclose(input_,ipa_)
            norm_input = input_align / (input_align.norm(dim=1,keepdim=True))
            norm_ipa = ipa_align / (ipa_align.norm(dim=1,keepdim=True))
            alignment = norm_input @ norm_ipa.transpose(1,2)
            alignment = self.logit_scale * (alignment)
                     
            assert torch.isnan(alignment).all() == False    
            align_loss = self.criterion_r(alignment,align_labels)
            align_loss_t = self.criterion_r(alignment.transpose(1,2),align_labels)
                     
            full_align = (align_loss + align_loss_t)/ 2
    return full_align

def update_pretrained_weights(args, model, lm, config):
    if (args.use_ipa == 'true'):
            if ('bert' in args.model_type):        
                    new_input_embs = torch.zeros(model.bert.embeddings.word_embeddings.weight.shape)
                    new_input_embs = nn.init.xavier_uniform_(new_input_embs)
                    #print ("Check sum", new_input_embs.sum(dim=-1))
                    model.bert.embeddings.ipa_embeddings.weight = torch.nn.Parameter(new_input_embs)

                    # Handle for MLM resizing
                    if (args.add_mlm == 'true'):
                            #Reshaping (extend vocab)
                            #NOTE: Same shape: predictions.transform.dense.bias, transform.LayerNorm.weight/bias [768]
                            #NOTE: Shape: decoder: vocab_size, 768, bias: vocab_size

                            print ("--Add MLM ---")
                            model.cls.predictions.transform.dense.weight = torch.nn.Parameter(lm['predictions.transform.dense.weight'])
                            model.cls.predictions.transform.dense.bias = torch.nn.Parameter(lm['predictions.transform.dense.bias'])

                            model.cls.predictions.transform.LayerNorm.weight = torch.nn.Parameter(lm['predictions.transform.LayerNorm.weight'])
                            model.cls.predictions.transform.LayerNorm.bias = torch.nn.Parameter(lm['predictions.transform.LayerNorm.bias'])

                            
                            decoder_weight = torch.zeros((config.vocab_size,768))
                            decoder_weight = nn.init.xavier_uniform_(decoder_weight)
                                    
                            bias_weight = torch.zeros((config.vocab_size,))
                            bias_weight.fill_(0.01)

                            model.cls.predictions.decoder.weight = torch.nn.Parameter(decoder_weight)
                            model.cls.predictions.decoder.bias = torch.nn.Parameter(bias_weight)
                            model.cls.predictions.bias = torch.nn.Parameter(bias_weight)

                    if (args.add_xmlm == 'true'):
                            print ("--Add X-MLM ---")
                            model.xcls.predictions.transform.dense.weight = torch.nn.Parameter(lm['predictions.transform.dense.weight'])
                            model.xcls.predictions.transform.dense.bias = torch.nn.Parameter(lm['predictions.transform.dense.bias'])


                            model.xcls.predictions.transform.LayerNorm.weight = torch.nn.Parameter(lm['predictions.transform.LayerNorm.weight'])
                            model.xcls.predictions.transform.LayerNorm.bias = torch.nn.Parameter(lm['predictions.transform.LayerNorm.bias'])

                            
                            decoder_weight = torch.zeros((config.vocab_size,768))
                            decoder_weight = nn.init.xavier_uniform_(decoder_weight)

                            bias_weight = torch.zeros((config.vocab_size,))
                            bias_weight.fill_(0.01)

                            model.xcls.predictions.decoder.weight = torch.nn.Parameter(decoder_weight)
                            model.xcls.predictions.decoder.bias = torch.nn.Parameter(bias_weight)
                            model.xcls.predictions.bias = torch.nn.Parameter(bias_weight)

            else:
                    new_input_embs = torch.zeros(model.roberta.embeddings.word_embeddings.weight.shape)
                    new_input_embs = nn.init.xavier_uniform_(new_input_embs)
                    #print ("Check sum", new_input_embs.sum(dim=-1))
                    model.roberta.embeddings.ipa_embeddings.weight = torch.nn.Parameter(new_input_embs)

                    if (args.add_mlm == 'true'):

                            print ("--Add MLM ---")
                            model.cls.dense.weight = torch.nn.Parameter(lm['dense.weight'])
                            model.cls.dense.bias = torch.nn.Parameter(lm['dense.bias'])

                            model.cls.layer_norm.weight = torch.nn.Parameter(lm['layer_norm.weight'])
                            model.cls.layer_norm.bias = torch.nn.Parameter(lm['layer_norm.bias'])

                            
                            decoder_weight = torch.zeros((config.vocab_size,768)) #XLM-R Large Model
                            decoder_weight = nn.init.xavier_uniform_(decoder_weight)
                                    
                            bias_weight = torch.zeros((config.vocab_size,))
                            bias_weight.fill_(0.01)

                            model.cls.decoder.weight = torch.nn.Parameter(decoder_weight)
                            model.cls.decoder.bias = torch.nn.Parameter(bias_weight)
                            model.cls.bias = torch.nn.Parameter(bias_weight)


                    if (args.add_xmlm == 'true'):

                            print ("--Add X-MLM ---")
                            model.xcls.dense.weight = torch.nn.Parameter(lm['dense.weight'])
                            model.xcls.dense.bias = torch.nn.Parameter(lm['dense.bias'])

                            model.xcls.layer_norm.weight = torch.nn.Parameter(lm['layer_norm.weight'])
                            model.xcls.layer_norm.bias = torch.nn.Parameter(lm['layer_norm.bias'])

                            
                            decoder_weight = torch.zeros((config.vocab_size,768)) #XLM-R Large Model
                            decoder_weight = nn.init.xavier_uniform_(decoder_weight)

                            bias_weight = torch.zeros((config.vocab_size,))
                            bias_weight.fill_(0.01)

                            model.xcls.decoder.weight = torch.nn.Parameter(decoder_weight)
                            model.xcls.decoder.bias = torch.nn.Parameter(bias_weight)
                            model.xcls.bias = torch.nn.Parameter(bias_weight)

    return model

	

def obtain_ipa_chars():
	ipa_file2 = '../ipa/ipa_symbols.txt'
	ipa_file1 = '../ipa/ipa_tok_symbols.txt'

	#print ("Obtain char", ipa_file1, ipa_file2)
	#ipa_file2='/hdd/multi_project/ipa_symbols.txt'
	#ipa_file1='/hdd/multi_project/ipa_tok_symbols.txt'
	ipa_r1 = open(ipa_file1, 'r')
	ipa_r2 = open(ipa_file2,'r')
	ipa_chars=set()
	for line in ipa_r1:
		line = line.strip()
		#symbol, dec_code, hex_code, descrp = line.split('\t')

		symbol = line.split('\t')[0]
		symbol = symbol.strip()
		ipa_chars.add(symbol)

	for line in ipa_r2:
		line = line.strip()
		symbol = line.split('\t')[0]
		symbol = symbol.strip()
		ipa_chars.add(symbol)
	ipa_chars = sorted(list(ipa_chars))
	#print ("IPA Chars", ipa_chars)
	return ipa_chars

# Can be used for both MLM and XMLM calculation
def update_mlm_for_acc(mlm_logits, ori_mlm_mask, all_mlm_indices, all_mlm_inputs, all_mlm_ids, all_mlm_preds, all_top_mlm_preds, all_top_mlm_scores,all_inputs, inputs):
    mlm_mask = torch.where(ori_mlm_mask ==-100, 0,1)
    mlm_final = mlm_logits * mlm_mask.unsqueeze(-1) #[bsz, max_seq_len, vocab_size] 
    values, indices = torch.max(mlm_final, dim=-1) # Output should be [8,128] 
                                                                                                                                                                                                                                                                             
    mlm_preds = np.argmax(mlm_logits.detach().cpu().numpy(), axis=2)
    top_mlm_score, top_mlm_pred = torch.topk(mlm_logits.detach().cpu(), 10, dim=2) #[16,128,5]

    if all_mlm_indices is None:
            all_mlm_indices = indices.detach().cpu().numpy()
            all_inputs = inputs['input_ids'].detach().cpu().numpy()
            all_mlm_inputs = inputs['mlm_input'].detach().cpu().numpy()
            all_mlm_ids = inputs['mlm_labels'].detach().cpu().numpy()
            #all_mlm_preds = mlm_logits.detach().cpu().numpy()

            all_mlm_preds = mlm_preds
            all_top_mlm_preds = top_mlm_pred
            all_top_mlm_scores = top_mlm_score

    else:
            all_mlm_indices  = np.append(all_mlm_indices, indices.detach().cpu().numpy(), axis=0)
            all_inputs      = np.append(all_inputs, inputs['input_ids'].detach().cpu().numpy(), axis=0)
            all_mlm_inputs  = np.append(all_mlm_inputs, inputs['mlm_input'].detach().cpu().numpy(), axis=0)
            all_mlm_ids  = np.append(all_mlm_ids, inputs['mlm_labels'].detach().cpu().numpy(), axis=0)

            #all_mlm_preds = np.append(all_mlm_preds, mlm_logits.detach().cpu().numpy(), axis=0)
            all_mlm_preds = np.append(all_mlm_preds, mlm_preds, axis=0)
            all_top_mlm_preds = np.append(all_top_mlm_preds, top_mlm_pred, axis=0)
            all_top_mlm_scores = np.append(all_top_mlm_scores, top_mlm_score, axis=0)

    return all_mlm_indices, all_inputs, all_mlm_inputs, all_mlm_ids, all_mlm_preds, all_top_mlm_preds, all_top_mlm_scores

def calculate_mlm_acc(label_map, pad_token_label_id, mlm_preds, all_mlm_ids, all_top_mlm_preds,all_top_mlm_scores):
    
    out_label_list = [[] for _ in range(mlm_preds.shape[0])]
    preds_list = [[] for _ in range(mlm_preds.shape[0])]    
                                                            
    top_preds_list = [[] for _ in range(mlm_preds.shape[0])]
                                                            
    acc = []                                                
    for i in range(mlm_preds.shape[0]):                     
            cur_acc=0                                       
            counter = 0                                     
            for j in range(mlm_preds.shape[1]):             
                    if all_mlm_ids[i, j] != pad_token_label_id:
                            out_label_list[i].append(label_map[all_mlm_ids[i][j]])
                            preds_list[i].append(label_map[mlm_preds[i][j]])     
                            #top_k=[]                       
                            top_selection = all_top_mlm_preds[i][j] #[16,128,5] => [5,]
                            top_scores = all_top_mlm_scores[i][j]          
                            top_k=[label_map[top_selection[k].item()] for k in range (top_selection.shape[-1])]
                            if (mlm_preds[i][j] == all_mlm_ids[i][j]):
                                            cur_acc += 1    
                            counter += 1                    
                    cur_acc  = cur_acc / (counter + 1e-20)  
                    acc.append(cur_acc)                     
                                                            
    return np.array(acc).mean()

