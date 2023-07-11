# codin=gutf-8
# Copyright 2018 The Google AI Language Team Authors,
# The HuggingFace Inc. team, and The XTREME Benchmark Authors.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#			http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning models for NER and POS tagging."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import copy
import math
import subprocess

import numpy as np
import torch
import torch.nn as nn
from seqeval.metrics import precision_score, recall_score, f1_score
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from utils_tag import convert_examples_to_features
from utils_tag import get_labels
from utils_tag import read_examples_from_file
import utils

from transformers import (
	AdamW,
	get_linear_schedule_with_warmup,
	WEIGHTS_NAME,
	BertConfig,
	BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
	BertTokenizer,
	BertForTokenClassification,
	BertForMaskedLM,
	XLMConfig,
	XLM_PRETRAINED_CONFIG_ARCHIVE_MAP,
	XLMTokenizer,
	XLMRobertaConfig,
	XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
	XLMRobertaTokenizer,
	XLMRobertaForTokenClassification,
	XLMRobertaForMaskedLM
)
from xlm import XLMForTokenClassification


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
	(tuple(conf.keys())
		for conf in (BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, XLM_PRETRAINED_CONFIG_ARCHIVE_MAP, XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP)),
	()
)

MODEL_CLASSES = {
	"bert": (BertConfig, BertForTokenClassification, BertTokenizer, BertForMaskedLM),
	"xlm": (XLMConfig, XLMForTokenClassification, XLMTokenizer),
	"xlmr": (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer, XLMRobertaForMaskedLM),
}

#--- Show GPU usage for debugging -----#
def show_gpu(msg):
		"""
		ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
		"""
		def query(field):
						return(subprocess.check_output(
										['nvidia-smi', f'--query-gpu={field}',
														'--format=csv,nounits,noheader'],
										encoding='utf-8'))
		def to_int(result):
						return int(result.strip().split('\n')[0])

		used = to_int(query('memory.used'))
		total = to_int(query('memory.total'))
		pct = used/total
		print('\n' + msg, f'{100*pct:2.1f}% ({used} out of {total})')


def set_seed(seed,n_gpu):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if n_gpu > 0:
		torch.cuda.manual_seed_all(seed)

	torch.backends.cudnn.determistic=True
	torch.backends.cudnn.benchmark=False
	os.environ['PYTHONHASHSEED'] = str(seed)
	if (torch.cuda.is_available()):
		os.environ['CUDA_LAUNCH_BLOCKING']='1'
		os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
		os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7,8'

def train(seed_val, word_dict, ipa_dict, args, train_dataset, model, tokenizer, labels, pad_token_label_id, lang2id=None):
	"""Train the model."""
	if args.local_rank in [-1, 0]:
		tb_writer = SummaryWriter()

	args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
	train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
	train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

	if args.max_steps > 0:
		t_total = args.max_steps
		args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
	else:
		t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

	# Prepare optimizer and schedule (linear warmup and decay)
	no_decay = ["bias", "LayerNorm.weight"] #no decay for bias and layernorm weight
	#optimizer_grouped_parameters = [
	#	{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'ipa_embeddings' not in n],
	#	 "weight_decay": args.weight_decay},
	#	{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) ], "weight_decay": 0.0},
	#	{'params':[p for n,p in model.named_parameters() if ('ipa_embeddings' in n)], 'lr': 2e-6, 'weight_decay': args.weight_decay}
	#]


	optimizer_grouped_parameters = [
		{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
		 "weight_decay": args.weight_decay},
		{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) ], "weight_decay": 0.0},
	]
	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
	if args.fp16:
		try:
			from apex import amp
		except ImportError:
			raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
		model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

	# multi-gpu training (should be after apex fp16 initialization)
	if args.n_gpu > 1:
		model = torch.nn.DataParallel(model)

	# Distributed training (should be after apex fp16 initialization)
	if args.local_rank != -1:
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
															output_device=args.local_rank,
															find_unused_parameters=True)

	# Train!
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(train_dataset))
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
	logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
				args.train_batch_size * args.gradient_accumulation_steps * (
					torch.distributed.get_world_size() if args.local_rank != -1 else 1))
	logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
	logger.info("  Total optimization steps = %d", t_total)

	best_score = 0.0
	best_checkpoint = None
	patience = 0
	global_step = 0
	tr_loss, logging_loss = 0.0, 0.0
	model.zero_grad()
	train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
	set_seed(seed_val, args.n_gpu) # Add here for reproductibility (even between python 2 and 3)
	if (args.device != -1):
		torch.cuda.empty_cache()

	for _ in train_iterator:
		epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
		for step, batch in enumerate(epoch_iterator):
			model.train()
			batch = tuple(t.to(args.device) for t in batch if t is not None)

			inputs = {"input_ids": batch[0],
						"attention_mask": batch[1],
						"input_emb_mask": batch[5],
						"labels": batch[3]}
			if (args.use_lang == 'true'):
				inputs.update({"lang_ids": batch[12]})
			if (args.use_ipa == 'true'):
				inputs.update({
					"ipa_ids":batch[8],
					"ipa_emb_mask": batch[6],
					"ipa_features": batch[11],
				})
				if (args.add_mlm== 'true'):
                                        inputs.update({	
                                                'mlm_input': batch[7],
                                                'mlm_ipa': batch[8],
                                                'mlm_labels': batch[9],
                                                })


				if (args.add_xmlm=='true'):
					inputs.update({	
						'xmlm_input': batch[13],
						'xmlm_ipa': batch[14],
						'xmlm_labels': batch[15],
						'xmlm_lang_ids':batch[17],
						})

			if args.model_type != "distilbert":
				# XLM and RoBERTa don"t use segment_ids
				inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None

			if args.model_type == "xlm":
				inputs["langs"] = batch[4]

			outputs = model(**inputs)
			loss = outputs[0]
			
			if args.n_gpu > 1:
				# mean() to average on multi-gpu parallel training
				loss = loss.mean()
			if args.gradient_accumulation_steps > 1:
				loss = loss / args.gradient_accumulation_steps

			if args.fp16:
				with amp.scale_loss(loss, optimizer) as scaled_loss:
					scaled_loss.backward()
			else:
				loss.backward()

			tr_loss += loss.item()


			if (step + 1) % args.gradient_accumulation_steps == 0:
				#0.7%
				if args.fp16:
					torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
				else:
					torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

				scheduler.step()	# Update learning rate schedule
				optimizer.step()
				model.zero_grad()
				global_step += 1
				if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
					# Log metrics
					if args.local_rank == -1 and args.evaluate_during_training:
						# Only evaluate on single GPU otherwise metrics may not average well
						results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", lang=args.train_langs, lang2id=lang2id)
						for key, value in results.items():
							tb_writer.add_scalar("eval_{}".format(key), value, global_step)
					tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
					tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
					logging_loss = tr_loss

				if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:

					#show_gpu("Save steps")
					if args.save_only_best_checkpoint:
						result, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", prefix=global_step, lang=args.train_langs, word_dict=word_dict, ipa_dict=ipa_dict, lang2id=lang2id,noloss=True)
						if result["f1"] > best_score:
							logger.info("result['f1']={} > best_score={}".format(result["f1"], best_score))
							best_score = result["f1"]
							# Save the best model checkpoint
							output_dir = os.path.join(args.output_dir, "checkpoint-best")
							best_checkpoint = output_dir
							if not os.path.exists(output_dir):
								os.makedirs(output_dir)
							# Take care of distributed/parallel training
							model_to_save = model.module if hasattr(model, "module") else model
							model_to_save.save_pretrained(output_dir)
							torch.save(args, os.path.join(output_dir, "training_args.bin"))
							logger.info("Saving the best model checkpoint to %s", output_dir)
							logger.info("Reset patience to 0")
							patience = 0
							#break #for testing only
						else:
							patience += 1
							logger.info("Hit patience={}".format(patience))
							if args.eval_patience > 0 and patience > args.eval_patience:
								logger.info("early stop! patience={}".format(patience))
								epoch_iterator.close()
								train_iterator.close()
								if args.local_rank in [-1, 0]:
									tb_writer.close()
								return global_step, tr_loss / global_step
					else:
						# Save model checkpoint
						output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
						if not os.path.exists(output_dir):
							os.makedirs(output_dir)
						# Take care of distributed/parallel training
						model_to_save = model.module if hasattr(model, "module") else model
						model_to_save.save_pretrained(output_dir)
						torch.save(args, os.path.join(output_dir, "training_args.bin"))
						logger.info("Saving model checkpoint to %s", output_dir)

						

					#show_gpu("End saving steps")

			if args.max_steps > 0 and global_step > args.max_steps:
				epoch_iterator.close()
				break

			#show_gpu("One training iterator")
		if args.max_steps > 0 and global_step > args.max_steps:
			train_iterator.close()
			break

	if args.local_rank in [-1, 0]:
		tb_writer.close()

	return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix="", lang="en", word_dict=None, ipa_dict=None,pretrain=False,lang2id=None, print_result=True, noloss=False):
        eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode, lang=lang, word_dict=word_dict,ipa_dict=ipa_dict, lang2id=lang2id)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu evaluate
        if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)

        # Eval!

        logger.info("***** Running evaluation %s in %s (mode:%s) *****" % (prefix, lang, mode))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        all_mlm_indices, all_inputs= None, None
        all_mlm_inputs, all_xmlm_inputs= None, None
        all_mlm_ids, all_xmlm_ids = None, None
        all_mlm_preds, all_xmlm_preds = None, None
        all_top_mlm_preds, all_top_mlm_scores = None, None

        all_top_xmlm_preds, all_top_xmlm_scores = None, None
        all_xmlm_indices = None
        rep_input=[]
        model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = tuple(t.to(args.device) for t in batch)
                with torch.no_grad():
                        inputs = {"input_ids": batch[0],
                                                "attention_mask": batch[1],
                                                "input_emb_mask": batch[5],
                                        }

                        if (args.use_lang == 'true'):
                                inputs.update({"lang_ids": batch[12]})
                        if (args.use_ipa == 'true'):
                                inputs.update({
                                        "ipa_ids":batch[8],
                                        "ipa_emb_mask": batch[6],
                                        "ipa_features": batch[11],
                                })
                                if (args.add_mlm== 'true'):

                                        inputs.update({	
                                                'mlm_input': batch[7],
                                                'mlm_ipa': batch[8],
                                                })

                                if (args.add_xmlm=='true'):
                                        inputs.update({	
                                                'xmlm_input': batch[13],
                                                'xmlm_ipa': batch[14],
                                                'xmlm_lang_ids':batch[17],
                                                })

                        inputs.update({
                                        "labels": batch[3]
                                })
                        if (noloss == False):
                                if (args.add_xmlm == 'true'):
                                        inputs.update({
                                                'xmlm_labels': batch[15],
                                                })
                        
                                if (args.add_mlm == 'true'):

                                        inputs.update({	
                                                'mlm_labels': batch[9],
                                                })
                                
                        if args.model_type != "distilbert":
                                # XLM and RoBERTa don"t use segment_ids
                                inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None
                        if args.model_type == 'xlm':
                                inputs["langs"] = batch[4]
                        outputs = model(**inputs)

                        #---- Computing loss during training ----#
#                        if (noloss == False):
#                                if (args.add_mlm == 'true'):
#                                    mlm_logits = outputs.mlm_logits
#                                    mlm_mask = batch[9]
#                                    all_mlm_indices, all_inputs, all_mlm_inputs, all_mlm_ids, all_mlm_preds, all_top_mlm_preds, all_top_mlm_scores = utils.update_mlm_for_acc(mlm_logits, mlm_mask, all_mlm_indices, all_mlm_inputs, all_mlm_ids, all_mlm_preds, all_top_mlm_preds, all_top_mlm_scores, all_inputs, inputs)
#
#                                if (args.add_xmlm == 'true'):
#                                    xmlm_logits = outputs.xmlm_logits
#                                    xmlm_mask = batch[15]
#
#                                    all_xmlm_indices, all_inputs, all_xmlm_inputs, all_xmlm_ids, all_xmlm_preds, all_top_xmlm_preds, all_top_xmlm_scores = utils.update_mlm_for_acc(xmlm_logits, xmlm_mask, all_xmlm_indices, all_xmlm_inputs, all_xmlm_ids, all_xmlm_preds, all_top_xmlm_preds, all_top_xmlm_scores,all_inputs, inputs)
#
                        if (noloss == False):
                                tmp_eval_loss, logits = outputs[:2]

                                if args.n_gpu > 1:
                                        # mean() to average on multi-gpu parallel evaluating
                                        tmp_eval_loss = tmp_eval_loss.mean()

                                eval_loss += tmp_eval_loss.item()
                        else:
                                tmp_eval_loss = 0.0
                                logits = outputs['logits']
                                eval_loss = 0.0


                nb_eval_steps += 1
                if (pretrain == False):
                        if preds is None:
                                preds = logits.detach().cpu().numpy() #[16,128,7], [16,128]
                                out_label_ids = inputs["labels"].detach().cpu().numpy()
                        else:
                                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                
                #show_gpu("Batch eval")
        if nb_eval_steps == 0:
                results = {k: 0 for k in ["loss", "precision", "recall", "f1"]}
        else:
                eval_loss = eval_loss / nb_eval_steps
                
                if (pretrain == True):
                        results = {
                                "loss":eval_loss,
                                }
                        preds_list = None
                else:
                        
                        #Sequence-labelling Eval
                        preds = np.argmax(preds, axis=2) #[16,128]
                        
                        label_map = {i: label for i, label in enumerate(labels)} #{0:B-person, 1:I-person}

                        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
                        preds_list = [[] for _ in range(out_label_ids.shape[0])]

                        for i in range(out_label_ids.shape[0]):
                                for j in range(out_label_ids.shape[1]):
                                        if out_label_ids[i, j] != pad_token_label_id:
                                                out_label_list[i].append(label_map[out_label_ids[i][j]])
                                                preds_list[i].append(label_map[preds[i][j]])

                        results = {
                                "loss": eval_loss,
                                "precision": precision_score(out_label_list, preds_list),
                                "recall": recall_score(out_label_list, preds_list),
                                "f1": f1_score(out_label_list, preds_list)
                        }
        if (noloss == False): # only for DEV sets
                label_map = {v: k for k, v in tokenizer.get_vocab().items()}

                if (args.add_mlm == 'true' and args.add_xmlm == 'true'	and lang != 'zh'):
                        for idx in range (all_mlm_indices.shape[0]):
                                inp= all_inputs[idx]
                                mlm_id = all_mlm_indices[idx]
                                xmlm_id = all_xmlm_indices[idx]
                                tokens = tokenizer.convert_ids_to_tokens(list(inp))
                                tokens = [t for t in tokens if t !='[PAD]']
                                mlm_tokens = tokenizer.convert_ids_to_tokens(list(mlm_id))
                                mlm_tokens = mlm_tokens[: len(tokens)]
                                xmlm_tokens = tokenizer.convert_ids_to_tokens(list(xmlm_id))

                                xmlm_tokens = xmlm_tokens[: len(tokens)]


        if print_result:

                logger.info("***** Evaluation result %s in %s *****" % (prefix, lang))
                for key in sorted(results.keys()):
                        logger.info("  %s = %s", key, str(results[key]))

        return results, preds_list


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode, lang, word_dict, ipa_dict,  lang2id=None, few_shot=-1):
	# Make sure only the first process in distributed training process
	# the dataset, and the others will use the cache
	if args.local_rank not in [-1, 0] and not evaluate:
		torch.distributed.barrier()

	# Load data features from cache or dataset file
	cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}_{}".format(mode, lang,
		list(filter(None, args.model_name_or_path.split("/"))).pop(),
		str(args.max_seq_length)))
	if os.path.exists(cached_features_file) and not args.overwrite_cache:
		logger.info("Loading features from cached file %s", cached_features_file)
		features = torch.load(cached_features_file)
	else:
	#if (True):
		langs = lang.split(',')
		logger.info("all languages = {}".format(lang))
		lang_2id={args.train_langs:0, args.predict_langs:1}
		print ("lang2id", lang_2id)
		features = []
		for lg in langs:
			data_file = os.path.join(args.data_dir, lg, "{}.{}".format(mode, args.model_name_or_path))

			print ("Data file", data_file)
			ipa_data_file = os.path.join(args.data_dir, lg, "{}.{}".format(mode+'-ipa', args.model_name_or_path))
			logger.info("Creating features from dataset file at {} in language {}".format(data_file, lg))
			examples = read_examples_from_file(args, data_file, lg, lang2id)

			features_lg = convert_examples_to_features(args,mode, examples, labels, args.max_seq_length, tokenizer, word_dict, ipa_dict,
													cls_token_at_end=bool(args.model_type in ["xlnet"]),
													cls_token=tokenizer.cls_token,
													cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
													sep_token=tokenizer.sep_token,
													sep_token_extra=bool(args.model_type in ["roberta", "xlmr"]),
													pad_on_left=bool(args.model_type in ["xlnet"]),
													pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
													pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
													pad_token_label_id=pad_token_label_id,
													lang=lg,
													langid=lang_2id
													)
			features.extend(features_lg)
		if args.local_rank in [-1, 0]:
			logger.info("Saving features into cached file {}, len(features)={}".format(cached_features_file, len(features)))
			torch.save(features, cached_features_file)

	print ("---Done reading ---")
	# Make sure only the first process in distributed training process
	# the dataset, and the others will use the cache
	if args.local_rank == 0 and not evaluate:
		torch.distributed.barrier()

	if few_shot > 0 and mode == 'train':
		logger.info("Original no. of examples = {}".format(len(features)))
		features = features[: few_shot]
		logger.info('Using few-shot learning on {} examples'.format(len(features)))

	# Convert to Tensors and build dataset
	all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
	all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
	all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
	all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)


	all_lang_ids = torch.tensor([f.normal_lang_ids for f in features], dtype=torch.long)

	all_xmlm_lang_ids = torch.tensor([f.lang_ids for f in features], dtype=torch.long)

	all_ipa_input_ids = torch.tensor([f.ipa_input_ids for f in features], dtype=torch.long)


	all_ipa_emb_mask = torch.tensor([f.ipa_emb_mask for f in features], dtype=torch.long)
	all_input_emb_mask = torch.tensor([f.input_emb_mask for f in features], dtype=torch.long)


	all_mlm_input_ids = torch.tensor([f.mlm_ids for f in features], dtype=torch.long)
	all_mlm_ipa_ids = torch.tensor([f.mlm_ipa_ids for f in features], dtype=torch.long)

	all_mlm_label = torch.tensor([f.mlm_label for f in features], dtype=torch.long)
	all_mlm_ipa_label = torch.tensor([f.mlm_ipa_label for f in features], dtype=torch.long)


	all_xmlm_input_ids = torch.tensor([f.xmlm_ids for f in features], dtype=torch.long)
	all_xmlm_ipa_ids = torch.tensor([f.xmlm_ipa_ids for f in features], dtype=torch.long)
	all_xmlm_label = torch.tensor([f.xmlm_label for f in features], dtype=torch.long)
	all_xmlm_ipa_label = torch.tensor([f.xmlm_ipa_label for f in features], dtype=torch.long)


	all_ipa_features = [f.ipa_features for f in features]
	all_ipa_features = torch.tensor(np.concatenate((all_ipa_features),0),dtype=torch.long)


	all_ipa_input_ids = torch.tensor([f.ipa_input_ids for f in features], dtype=torch.long)
	#all_ipa_features = torch.tensor(f.ipa_features. for f in features], dtype

	if args.model_type == 'xlm' and features[0].langs is not None:
		all_langs = torch.tensor([f.langs for f in features], dtype=torch.long)
		logger.info('all_langs[0] = {}'.format(all_langs[0]))
		dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_ipa_input_ids, all_input_emb_mask, all_ipa_emb_mask, all_langs)
	else:
		dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_ipa_input_ids, all_input_emb_mask, all_ipa_emb_mask,all_mlm_input_ids, all_mlm_ipa_ids, all_mlm_label,all_mlm_ipa_label, all_ipa_features,all_lang_ids,all_xmlm_input_ids, all_xmlm_ipa_ids, all_xmlm_label, all_xmlm_ipa_label,all_xmlm_lang_ids)
	return dataset


def load_updated_dictionary(args):

	dict_path = '../' + args.train_langs + '-' + args.predict_langs + '-dict.txt'
	print ("DIctionary file", dict_path)
	word_dict, inv_word_dict={},{}
	ipa_dict, inv_ipa_dict ={}, {}
	for line in open(dict_path, 'r'):
		elements = line.strip().split('\t') #src_ori,src_ipa, tgt_ori, tgt_ipa

		src_word_corr, tgt_word_corr = elements[0],elements[2]
		src_ipa_corr, tgt_ipa_corr = elements[1], elements[3]
	
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


def main():
        parser = argparse.ArgumentParser()

        ## Required parameters
        parser.add_argument("--data_dir", default=None, type=str, required=True,
                                                help="The input data dir. Should contain the training files for the NER/POS task.")
        parser.add_argument("--model_type", default=None, type=str, required=True,
                                                help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
        parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                                                help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
        parser.add_argument("--output_dir", default=None, type=str, required=True,
                                                help="The output directory where the model predictions and checkpoints will be written.")

        ## Other parameters
        parser.add_argument("--labels", default="", type=str,
                                                help="Path to a file containing all labels. If not specified, NER/POS labels are used.")

        parser.add_argument("--config_name", default="", type=str,
                                                help="Pretrained config name or path if not the same as model_name")
        parser.add_argument("--tokenizer_name", default="", type=str,
                                                help="Pretrained tokenizer name or path if not the same as model_name")
        parser.add_argument("--cache_dir", default=None, type=str,
                                                help="Where do you want to store the pre-trained models downloaded from s3")
        parser.add_argument("--max_seq_length", default=128, type=int,
                                                help="The maximum total input sequence length after tokenization. Sequences longer "
                                                         "than this will be truncated, sequences shorter will be padded.")


# Coefficients
        parser.add_argument("--do_eval", default='true', type=str,
                                                help="Whether to run eval on the dev set.")


        parser.add_argument("--do_train", default='true', type=str,
                                                help="Whether to run eval on the dev set.")


        parser.add_argument("--do_predict_dev", default='true', type=str,
                                                help="Whether to run eval on the dev set.")

        parser.add_argument("--do_predict", action="store_true",
                                                help="Whether to run predictions on the test set.")
        parser.add_argument("--init_checkpoint", default=None, type=str,
                                                help="initial checkpoint for train/predict")
        parser.add_argument("--evaluate_during_training", action="store_true",
                                                help="Whether to run evaluation during training at each logging step.")
        parser.add_argument("--do_lower_case", action="store_true",
                                                help="Set this flag if you are using an uncased model.")
        parser.add_argument("--few_shot", default=-1, type=int,
                                                help="num of few-shot exampes")

        parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                                                help="Batch size per GPU/CPU for training.")
        parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                                                help="Batch size per GPU/CPU for evaluation.")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                                                help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument("--learning_rate", default=5e-5, type=float,
                                                help="The initial learning rate for Adam.")
        parser.add_argument("--weight_decay", default=0.0, type=float,
                                                help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                                                help="Epsilon for Adam optimizer.")
        parser.add_argument("--max_grad_norm", default=1.0, type=float,
                                                help="Max gradient norm.")
        parser.add_argument("--num_train_epochs", default=3.0, type=float,
                                                help="Total number of training epochs to perform.")
        parser.add_argument("--max_steps", default=-1, type=int,
                                                help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
        parser.add_argument("--warmup_steps", default=0, type=int,
                                                help="Linear warmup over warmup_steps.")

        parser.add_argument("--logging_steps", type=int, default=50,
                                                help="Log every X updates steps.")
        parser.add_argument("--save_steps", type=int, default=50,
                                                help="Save checkpoint every X updates steps.")
        parser.add_argument("--save_only_best_checkpoint", action="store_true",
                                                help="Save only the best checkpoint during training")
        parser.add_argument("--eval_all_checkpoints", action="store_true",
                                                help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
        parser.add_argument("--no_cuda", action="store_true",
                                                help="Avoid using CUDA when available")
        parser.add_argument("--overwrite_output_dir", action="store_true",
                                                help="Overwrite the content of the output directory")
        parser.add_argument("--overwrite_cache", action="store_true",
                                                help="Overwrite the cached training and evaluation sets")
        parser.add_argument("--seed", type=str, default='42',
                                                help="random seed for initialization")

        parser.add_argument("--fp16", action="store_true",
                                                help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
        parser.add_argument("--fp16_opt_level", type=str, default="O1",
                                                help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                                                         "See details at https://nvidia.github.io/apex/amp.html")
        parser.add_argument("--local_rank", type=int, default=-1,
                                                help="For distributed training: local_rank")
        parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
        parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
        parser.add_argument("--predict_langs", type=str, default="en", help="prediction languages")
        parser.add_argument("--train_langs", default="en", type=str,
                                                help="The languages in the training sets.")
        parser.add_argument("--log_file", type=str, default=None, help="log file")
        parser.add_argument("--eval_patience", type=int, default=-1, help="wait N times of decreasing dev score before early stop during training") #-1 before

        parser.add_argument("--gpu_id", type=str, default='')


        #Logistics
        parser.add_argument("--use_cs", type=str, default='true')
        parser.add_argument("--expand_vocab", type=str, default='none')
        parser.add_argument("--use_ipa", type=str, default='false')

        # IPA Alignment
        parser.add_argument("--cls_alignment", default='false', type=str,
                                                help="Use full seq alignment")
        parser.add_argument("--add_alignment", default='false', type=str,
                                                help="Add Alignment Loss or not")
        parser.add_argument('--align_coeff', default=1.0, type=float)

        # Add MLM objectives
        parser.add_argument("--add_mlm", default='false', type=str,
                                                help="MLM training or not")
        parser.add_argument('--mlm_coeff', default=1.0, type=float)
        parser.add_argument('--mlm_ratio', default=0.15, type=float)

        # Add XLM objectives
        parser.add_argument("--add_xmlm", default='false', type=str,
                                                help="Cross-lingual MLM training or not")
        parser.add_argument('--cs_ratio', default=0.1, type=float)

        parser.add_argument('--xmlm_coeff', default=1.0, type=float)

        parser.add_argument("--use_lang", default='false', type=str,
                                                help="Use LANG embeddings or not")

        parser.add_argument("--use_mlm_ratio", default='false', type=str,
                                                help="Use ratio MLM")

        # Kept default
        parser.add_argument('--alter_ratio', default=0.5, type=float)

        parser.add_argument("--use_roman", type=str, default='false')

        parser.add_argument("--use_new_pooling", default='false', type=str,
                                                help="New pooling")

        args = parser.parse_args()

        if os.path.exists(args.output_dir) and os.listdir(
                        args.output_dir) and args.do_train == 'true' and not args.overwrite_output_dir:
                raise ValueError(
                        "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                                args.output_dir))

# Setup distant debugging if needed
        if args.server_ip and args.server_port:
                import ptvsd
                print("Waiting for debugger attach")
                ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
                ptvsd.wait_for_attach()

        if (args.gpu_id):
                os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
# Setup CUDA, GPU & distributed training
        if args.local_rank == -1 or args.no_cuda:
                device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

                #device = args.gpu_id
                args.n_gpu = torch.cuda.device_count()
                #args.n_gpu=1
                #device = 0
        else:
# Initializes the distributed backend which sychronizes nodes/GPUs
                torch.cuda.set_device(args.local_rank)
                device = torch.device("cuda", args.local_rank)
                torch.distributed.init_process_group(backend="nccl")
                args.n_gpu = 1
        args.device = device

# Setup logging
        logging.basicConfig(handlers = [logging.FileHandler(args.log_file), logging.StreamHandler()],
                                        format = '%(asctime)s - %(levelname)s - %(name)s -	 %(message)s',
                                        datefmt = '%m/%d/%Y %H:%M:%S',
                                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
        logging.info("Input args: %r" % args)
        logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                                         args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

# Set seed

        src_dev, src_test, tgt_dev, tgt_test = [],[],[],[]
        eval_langs = [args.train_langs] + args.predict_langs.split(',')
        report_dict={
                        'dev':{},
                        'test':{}}

        all_seed = args.seed.split(',')
        for seed_val in all_seed:
                seed_val = int(seed_val)
                set_seed(seed_val, args.n_gpu)

                # Prepare NER/POS task
                labels = get_labels(args.labels)
                num_labels = len(labels)
                # Use cross entropy ignore index as padding label id
                # so that only real label ids contribute to the loss later
                pad_token_label_id = CrossEntropyLoss().ignore_index

                # Load pretrained model and tokenizer
                # Make sure only the first process in distributed training loads model/vocab
                if args.local_rank not in [-1, 0]:
                        torch.distributed.barrier()

                args.model_type = args.model_type.lower()
                config_class, model_class, tokenizer_class,pretrain_model_class = MODEL_CLASSES[args.model_type]
                config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                                                                                num_labels=num_labels,
                                                                                                cache_dir=args.cache_dir if args.cache_dir else None)
                tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                                                                        do_lower_case=args.do_lower_case,
                                                                                                        cache_dir=args.cache_dir if args.cache_dir else None)

                #--- Updated config for Phonemic Integration
                config.update({'use_ipa':args.use_ipa})
                #config.update({'mlm_both':args.mlm_both})
                config.update({'cls_alignment':args.cls_alignment})


                config.update({'add_mlm':args.add_mlm})
                config.update({'add_xmlm':args.add_xmlm})
                config.update({'add_alignment':args.add_alignment})


                config.update({'align_coeff':args.align_coeff})
                config.update({'mlm_coeff':args.mlm_coeff})
                config.update({'xmlm_coeff':args.xmlm_coeff})

                config.update({'mlm_ratio':args.mlm_ratio})

                config.update({'use_lang':args.use_lang})
                config.update({'use_new_pooling':args.use_new_pooling})

                print ("Updated Ortho-Phonemic config", config)
                ipa_chars = utils.obtain_ipa_chars()

                # Make sure only the first process in distributed training loads model/vocab
                if args.init_checkpoint:
                        logger.info("loading from init_checkpoint={}".format(args.init_checkpoint))
                        model = model_class.from_pretrained(args.init_checkpoint,
                                                            config=config,
                                                            cache_dir=args.init_checkpoint)

                        pretrain_model = pretrain_model_class.from_pretrained(args.init_checkpoint,
                                                            config=config,
                                                            cache_dir=args.init_checkpoint)
                else:
                        logger.info("loading from cached model = {}".format(args.model_name_or_path))

                        model = model_class.from_pretrained(args.model_name_or_path,
                                                                                                from_tf=bool(".ckpt" in args.model_name_or_path),
                                                                                                config=config,
                                                                                                cache_dir=args.cache_dir if args.cache_dir else None)

                        with torch.no_grad():
                                #print ("Pretrain model class", pretrain_model_class)
                                pretrain_model = pretrain_model_class.from_pretrained(args.model_name_or_path,
                                                                                                        from_tf=bool(".ckpt" in args.model_name_or_path),
                                                                                                        config=config,
                                                                                                        cache_dir=args.cache_dir if args.cache_dir else None)
                lang2id = config.lang2id if args.model_type == "xlm" else None
                logger.info("Using lang2id = {}".format(lang2id))

                import csv
                out_file = open('../report_'+ args.train_langs + '.tsv', 'a')
                out_writer = csv.writer(out_file, delimiter='\t')


                if (args.do_train == 'true'):
                        if (not os.path.exists('../report_' + args.train_langs + '.tsv')):
                                print ("--Write header --")
                                header = ['Config', 'Add_Align','Add_MLM','Add_XMLM','align_coeff','mlm_coeff','xmlm_coeff','mlm_ratio','src_f1','tgt_f1'] 
                                out_writer.writerow(header)
		
                # Prepare pretrained LM vectors to copy as initialization
                if (config.add_mlm == 'true' or config.add_xmlm=='true'):
                        with torch.no_grad():
                                if ('bert' in args.model_type):
                                        lm_model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased', config=config)
                                else:

                                        lm_model = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-base', config=config)
                                lm_model.eval()
                                lm={}
                                for nnn,pp in lm_model.named_parameters():
                                        rem_part = nnn.split('.')[1:]
                                        name = '.'.join(rem_part)
                                        if (name not in lm):
                                                lm[name] = pp.cpu().detach() 
                                #print ("LM", lm.keys())
                else:
                        lm = None

		# --- Update VOCAB for IPA usage ---#
                if (args.expand_vocab == 'true'):
                        tokenizer.add_tokens(list(ipa_chars))
                        config.vocab_size = len(tokenizer)	
                        model.resize_token_embeddings(len(tokenizer))

                model = utils.update_pretrained_weights(args, model, lm, config)

                print ("--- Token Model ---")
                for n,p in model.named_parameters():
                        if (p.requires_grad):
                                if ('encoder' not in n):
                                        print (n, p.shape, p.sum())
                                if ('cls' in n):
                                        rem_name = n.split('.')[1:]
                                        full_rem = '.'.join(rem_name)

                if args.local_rank == 0:
                        torch.distributed.barrier()
                model = model.to(args.device)
                logger.info("Training/evaluation parameters %s", args)
                print ("Done params")

                ipa_dict, inv_ipa_dict, ipa_full_dict = None, None, None
                inv_word_dict=None

                if (args. add_xmlm == 'true'):
                        word_dict,_,ipa_full_dict, inv_ipa_full_dict =	load_updated_dictionary(args)
                else:
                        word_dict, ipa_full_dict = None, None

                # Training
                if args.do_train == 'true':
                        print ("Start training")
                        train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train", lang=args.train_langs, word_dict=ipa_full_dict, ipa_dict = ipa_dict, lang2id=lang2id, few_shot=args.few_shot)
                        print ("---- Real Training ----")
                        global_step, tr_loss = train(seed_val, word_dict,ipa_dict, args, train_dataset,model, tokenizer, labels, pad_token_label_id, lang2id)

                        print ("Training loss", tr_loss)
                        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

                # Saving best-practices: if you use default names for the model,
                # you can reload it using from_pretrained()
                if args.do_train == 'true' and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                        # Create output directory if needed
                        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                                os.makedirs(args.output_dir)

                        # Save model, configuration and tokenizer using `save_pretrained()`.
                        # They can then be reloaded using `from_pretrained()`
                        # Take care of distributed/parallel training
                        logger.info("Saving model checkpoint to %s", args.output_dir)
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(args.output_dir)
                        tokenizer.save_pretrained(args.output_dir)

                        # Good practice: save your training arguments together with the model
                        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

                # Initialization for evaluation
                results = {}
                if args.init_checkpoint:
                        best_checkpoint = args.init_checkpoint
                elif os.path.exists(os.path.join(args.output_dir, 'checkpoint-best')):
                        best_checkpoint = os.path.join(args.output_dir, 'checkpoint-best')
                else:
                        best_checkpoint = args.output_dir
                best_f1 = 0

                # Evaluation
                if args.do_eval == 'true' and args.local_rank in [-1, 0]:
                        #tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
                        checkpoints = [args.output_dir]
                        if args.eval_all_checkpoints:
                                checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
                                logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)
                        logger.info("Evaluate the following checkpoints: %s", checkpoints)

                        for checkpoint in checkpoints:
                                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                                model = model_class.from_pretrained(checkpoint)
                                model.to(args.device)
                                result, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", prefix=global_step, lang=args.train_langs, word_dict=ipa_full_dict, ipa_dict=None,lang2id=lang2id,noloss=False)

                                if result["f1"] > best_f1:
                                        best_checkpoint = checkpoint
                                        best_f1 = result["f1"]
                                if global_step:
                                        result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
                                results.update(result)

                        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                        with open(output_eval_file, "w") as writer:
                                for key in sorted(results.keys()):
                                        writer.write("{} = {}\n".format(key, str(results[key])))
                                writer.write("best checkpoint = {}, best f1 = {}\n".format(best_checkpoint, best_f1))
                        print ("Eval result", results)

		# Predict dev set
		# For testing, do not pass labels => do not need loss
                if args.do_predict_dev == 'true' and args.local_rank in [-1, 0]:
                        print ("---- Dev Result-----")
                        print ("Expand_Vocab :%s"%(args.expand_vocab))
                        print ("Source:%s - Tgt:%s"%(args.train_langs, args.predict_langs))
                        print ("Use IPA Embedding:%s"%(args.use_ipa))
                        print ("-----------------")
                        logger.info("Loading the best checkpoint from {}\n".format(best_checkpoint))
                        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

                        #Expand tokenizer vocab
                        
                        model = model_class.from_pretrained(best_checkpoint)
                        model.to(args.device)

                        output_test_results_file = os.path.join(args.output_dir, "dev_results.txt")
                        results={}
                        modes=['dev', 'test']
                        with open(output_test_results_file, "w") as result_writer:
                                eval_langs = [args.train_langs] + args.predict_langs.split(',')
                                for lang in eval_langs:
                                        for m in modes:
                                                if not os.path.exists(os.path.join(args.data_dir, lang, str(m) +'.{}'.format(args.model_name_or_path))):
                                                        logger.info("Language {} does not exist".format(lang))
                                                        continue
                                                if (lang == args.predict_langs.split(',')[0] and args.add_xmlm == 'true'):
                                                        result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode=m, lang=lang, word_dict = inv_ipa_full_dict, ipa_dict=None,lang2id=lang2id,noloss=True)
                                                else:
                                                        result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode=m, lang=lang, word_dict = ipa_full_dict, ipa_dict=None,lang2id=lang2id,noloss=True)
                                                if (lang in report_dict[m]):
                                                        report_dict[m][lang].append(result['f1'])
                                                else:

                                                        report_dict[m][lang]=[result['f1']]


                                                # Save results
                                                result_writer.write("=====================\nlanguage={}, mode={}\n".format(lang, m))
                                                for key in sorted(result.keys()):
                                                        result_writer.write("{}-{} = {}\n".format(key, m, str(result[key])))
                                                results[lang] = result
					## Save predictions
					#output_test_predictions_file = os.path.join(args.output_dir, "dev_{}_predictions.txt".format(lang))
					#infile = os.path.join(args.data_dir, lang, "dev.{}".format(args.model_name_or_path))
					#idxfile = infile + '.idx'
					#save_predictions(args, predictions, output_test_predictions_file, infile, idxfile)

					#print ("Predict-dev result", result)

        print ("***** Final Report *****")
        result_num=[]
        for mode in report_dict:
                lang_dict = report_dict[mode]
                for lang in lang_dict.keys():
                        result_array = np.array(lang_dict[lang])
                        mean, stdev = np.mean(result_array)*100.0, np.std(result_array)*100.0
                        print ("List", result_array)
                        print ("Lang: %s \t Type: %s \t Mean: %.8f \t STDEV: %.8f"%(str(lang), str(mode), mean, stdev))
                        result_num.append(tuple((mean,stdev, lang_dict[lang])))


        if (args.do_train == 'true' and args.do_predict_dev == 'true'):	
                out_writer.writerow([args, args.add_alignment, args.add_mlm, args.add_xmlm, args.align_coeff, args.mlm_coeff, args.xmlm_coeff, args.mlm_ratio, result_num])
	
def save_predictions(args, predictions, output_file, text_file, idx_file, output_word_prediction=False):
	# Save predictions
	with open(text_file, "r") as text_reader, open(idx_file, "r") as idx_reader:
		text = text_reader.readlines()
		index = idx_reader.readlines()
		assert len(text) == len(index)

	# Sanity check on the predictions
	with open(output_file, "w") as writer:
		example_id = 0
		prev_id = int(index[0])
		for line, idx in zip(text, index):
			if line == "" or line == "\n":
				example_id += 1
			else:
				cur_id = int(idx)
				output_line = '\n' if cur_id != prev_id else ''
				if output_word_prediction:
					output_line += line.split()[0] + '\t'
				output_line += predictions[example_id].pop(0) + '\n'
				writer.write(output_line)
				prev_id = cur_id

if __name__ == "__main__":
	main()
