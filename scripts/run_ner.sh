##----------- Generic Experiment Arguments ----------#
expand='true'
u_ipa='true'
lang='true'
use_roman='false'
lr=2e-5
maxl=128
 
align_signal='true'
mlm_signal='true'
xmlm_signal='true'
num_epoch=10
gpu_id=0
seeds='111,222,333'
 
#--- Parsing arguments --- #
model_name=${1-'mbert'}
src_lang=${2-'zh'}
tgt_lang=${3-'vi'}
 
#------- Specific Pretrain Type Arguments ------------#
if [ ${model_name} == "mbert" ]; then
    #BERT
    model='bert-base-multilingual-cased'
    b=8
    grad=4
    eval_batch=16
 
elif [ ${model_name} == 'xlmr' ]; then
    #XLM-R
    model='xlm-roberta-base'
    tok_t='xlm'
    b=2
    grad=8
    eval_batch=4
fi
  
#------ TASK SPECIFIC ARGUMENTS -------#
task='full_updated_panx'
save_step=1000
#------- Language Specific Arguments --------- #
if [ ${src_lang} == 'zh' ] && [ ${tgt_lang} == 'vi' ]; then
    #ZH-VI report
    src='zh'
    tgt='vi'
    align_coeff=0.01 #best
    mlm_coeff=0.01 #0.001 before
    xmlm_coeff=0.01 #best
    mlm_ratio=0.20
    cs_ratio=0.40
 
elif [ ${src_lang} == "ja" ] && [ ${tgt_lang} == "ko" ]; then
    #JA-KO (paper report)
    src='ja'
    tgt='ko'
    align_coeff=0.1
    mlm_coeff=0.001
    xmlm_coeff=0.001
    mlm_ratio=0.25
    cs_ratio=0.30
fi                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
 
#----- Run Preprocessing --------#
rm -r "../download/${task}/${task}_processed_maxlen${maxl}/${src}"
rm -r "../download/${task}/${task}_processed_maxlen${maxl}/${tgt}"
bash preprocess.sh ${task} ${expand} ${model} ${maxl} "${src},${tgt}"
 
#---- Preparing Arguments for main run -----#
args=""
args+="${src} " #src_lang (1)
args+="${tgt} " #tgt_lang (2)
args+="${gpu_id} " #gpu_id
args+="${model} " #xlmr or bert
args+="${task} " #task_type
args+="${maxl} " #max_length
args+="$b " #batch_size
args+="${eval_batch} " #eval_batch
args+="${grad} " #gradient accumulation
args+="${num_epoch} " # num_epoch training
args+="${save_step} " # save_step?   
args+="${seeds} " # seed_val?
args+="true " # train?
args+="true " # eval?
args+="true " # pred_dev?
 
 
# IPA Arguments
args+="${expand} " #expand vocab or not
args+="${u_ipa} " #use_ipa or not
 
 
#Unsupervised objectives
args+="${align_signal} " #add align
args+="${align_coeff} " # align coeff
 
args+="${mlm_signal} " # add mlm?
args+="${mlm_coeff} " # mlm coeff
 
args+="${xmlm_signal} " # add xmlm
args+="${xmlm_coeff} " # add coeff

  
# Extra arguments
args+="${lang} " # lang_emb
args+="${cs_ratio} " #code-switch ratio 
args+="${mlm_ratio} " # masking ratio
 
 
echo $args
#--- Run
bash train.sh $args
 
