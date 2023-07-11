#!/bin/bash
# Copyright 2020 Google and DeepMind.
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

#----- Pre-defined arguments ------#
REPO=$PWD
CODE_DIR='../'
use_roman='false'
DATA_DIR="../download/"
OUT_DIR="../outputs/"
PRETRAIN_DIR="none" #TODO: Future 2-stage process
LR=2e-5
alter_ratio=0.5
use_roman='false'
mlm_both='token'
use_mlm_ratio='false'


#------ Generic Experiment Arguments -------#
src_lang=${1:-"en"}
tgt_lang=${2:-"zh"}
GPU=${3:-"0"}
MODEL=${4-'xlm-roberta-base'}
TASK=${5:-'panx'}
MAX_LENGTH=${6:-128}
BATCH_SIZE=${7:-16}
eval_batch=${8:-32}
GRAD_ACC=${9:-4}

NUM_EPOCHS=${10:-10}
eval_step=${11:-1000}
seeds=${12:-'42'}
d_train=${13-'true'}
d_eval=${14-'true'}
d_pred_dev=${15-'true'}

echo $src_lang $tgt_lang

#------IPA-related arguments ------#
vocab=${16:-"false"}
use_ipa=${17:-"false"}


#---- Unsupervised objective arguments ----#

add_align=${18-'false'}
align_coeff=${19-0.01}

add_mlm=${20-'false'}
mlm_coeff=${21-0.01}

add_xmlm=${22-'false'}
xmlm_coeff=${23-0.01}

# Optional arguments
use_lang=${24-'false'}

cs_ratio=${25-0.5}
mlm_ratio=${26-0.5}

echo $TASK ${MODEL} ${cs_ratio}


export CUDA_VISIBLE_DEVICES=$GPU

LC=""
if [ $MODEL == "bert-base-multilingual-cased" ]; then
	MODEL_TYPE="bert"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
	MODEL_TYPE="xlmr"
fi


OUTPUT_DIR="$OUT_DIR/$TASK/${MODEL}-LR${LR}-epoch${NUM_EPOCHS}-MaxLen${MAX_LENGTH}-Src${src_lang}-Tgt${tgt_lang}-Batch${BATCH_SIZE}-EVAL${eval_batch}-full/" #normal model
DATA_DIR=$DATA_DIR/${TASK}/${TASK}_processed_maxlen${MAX_LENGTH}/
PRETRAIN_DIR="none"
mkdir -p $OUTPUT_DIR

python $CODE_DIR/third_party/run_tag.py \
	--data_dir $DATA_DIR \
	--model_type $MODEL_TYPE \
	--labels $DATA_DIR/labels.txt \
	--model_name_or_path $MODEL \
	--output_dir $OUTPUT_DIR \
	--max_seq_length $MAX_LENGTH \
	--num_train_epochs $NUM_EPOCHS \
	--gradient_accumulation_steps $GRAD_ACC \
	--per_gpu_train_batch_size $BATCH_SIZE \
	--per_gpu_eval_batch_size ${eval_batch} \
	--save_steps ${eval_step} \
	--learning_rate $LR \
	--use_mlm_ratio ${use_mlm_ratio} \
	--do_train ${d_train} \
	--do_eval ${d_eval} \
	--do_predict_dev ${d_pred_dev} \
	--predict_langs ${tgt_lang} \
	--train_langs ${src_lang} \
	--log_file $OUTPUT_DIR/train.log \
	--eval_patience 5 \
	--overwrite_cache \
	--overwrite_output_dir \
	--save_only_best_checkpoint $LC \
	--gpu_id ${GPU} \
	--expand_vocab ${vocab} \
	--use_ipa ${use_ipa} \
	--cls_alignment false \
	--add_alignment ${add_align} \
	--add_mlm ${add_mlm} \
	--add_xmlm ${add_xmlm} \
	--use_lang ${use_lang} \
	--align_coeff ${align_coeff} \
	--mlm_coeff ${mlm_coeff} \
	--xmlm_coeff ${xmlm_coeff} \
	--mlm_ratio ${mlm_ratio} \
	--cs_ratio ${cs_ratio} \
	--seed ${seeds} \
	--alter_ratio ${alter_ratio} \
	--use_roman ${use_roman} \

	#--mlm_both ${mlm_both} \
