#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

REPO=$PWD
#MODEL=${1:-bert-base-multilingual-cased}
#MODEL='bert-base-multilingual-cased'


DATA_DIR='../download/'
CODE_DIR='../'
TASK=${1-'panx'}
expand=${2-'true'}
MODEL=${3-'bert-base-multilingual-cased'}
MAXL=${4-128}

#echo $TASK
#echo $MODEL
#echo $MAXL
LANGS=${5:-'zh,vi'}
LC=""
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
  LC=" --do_lower_case"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlmr"
fi
SAVE_DIR="$DATA_DIR/$TASK/${TASK}_processed_maxlen${MAXL}"
mkdir -p $SAVE_DIR
python $CODE_DIR/utils_preprocess.py \
  --data_dir $DATA_DIR/$TASK/ \
  --task panx_tokenize \
  --model_name_or_path $MODEL \
  --model_type $MODEL_TYPE \
  --max_len $MAXL \
  --output_dir $SAVE_DIR \
  --languages $LANGS $LC \
  --expand_vocab ${expand}
  #--languages $LANGS $LC >> $SAVE_DIR/preprocess.log
echo "DONE"
if [ ! -f $SAVE_DIR/labels.txt ]; then
  cat $SAVE_DIR/*/*.${MODEL} | cut -f 2 | grep -v "^$" | sort | uniq > $SAVE_DIR/labels.txt
fi
