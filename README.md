# Enhancing Cross-lingual Transfer via Phonemic Transcription Integration (PhoneXL)
This repository provides PyTorch implementation for the paper [*Enhancing Cross-lingual Transfer via Phonemic Transcription Integration*](https://aclanthology.org/2023.findings-acl.583/) **(Findings of ACL'2023)**

## Requirements
Follow [XTREME benchmark](https://github.com/google-research/xtreme) requirements. Additional specific details of requriements are as follows:
python 3.9.12 <br />
numpy 1.22.3 <br />
pytorch 1.12.1 <br />
transformers 4.18.0 <br />
pandas 0.20.2 <br />
scikit-learn 1.0.2 <br />
seqeval 1.2.2 <br />
tqdm 4.64.1 <br />
tensorboardx 2.2 <br />


## Dataset
We generate and conduct evaluation on orthographic-phonemic alignment dataset for CJKV languages on Part-of-Speech (POS) and Named Entity Recognition (NER) tasks. The datasets are extracted from [XTREME benchmark](https://github.com/google-research/xtreme). Please take a look at our paper for details of the preprocessing steps.
 
## Configuration
Adjust configurations in ./scripts/run_ner.sh (for NER tasks) and ./scripts/run_pos.sh (for POS tasks)
<!-- Major important arguments are:

* ```--ckpt_dir```: Saved directory for checkpoint
* ```--eps```: Evaluate as nonepisodic or episodic procedure (i.e. eps or noneps)
* ```--num_eps```: Number of episodes used for episodic training and/or evaluation
* ```--dataset```: Choose dataset to train/evaluate (i.e. SNIPS/ NLUE)
* ```--num_run```: number of runs (only for SNIPS)
* ```--num_fold```: number of KFold counting from 1 to 10 (only for NLUE)
* ```--src```: Source data used for training (i.e. seen)
* ```--tgt```: Data used for evaluation (i.e. novel or joint)
* ```--num_samples_per_class```: K in C-way K-shot
* ```--num_class```: C class in C-way K-shot
* ```--num_query_per_class```: num query per class (Q)
* ```--num_test_class```: Number of classes used for evaluation (i.e. C for episodic, #total classes in joint/novel space)
* ```--fasttext_path```: FastText pretrained embedding file location

**Regularization hyperparameters**
* ```--self_attn_loss```: coefficient for Self-attention Regularization 
* ```--uniform_loss```: coefficient for Head Uniform Regularization
* ```--same_intent_loss```: coefficient for Head Distribution Regularization -->


## Running Experiments
Experiments are conducted on token-level NER and POS tasks for 2 language groups: ZH->VI, JA-> KO. 
Please refer to the manuscript regarding the detailed rationale of the experiment design.
```
bash run_${task}.sh ${backbone} ${source_lang} ${target_lang}
```
```

where passing arguments ${.} are defined as follows: 
* ```task``: Evaluation task (i.e. ner, pos)
* ```backbone```: Backbone PLMs (i.e. mbert, xlmr)
* ```source_lang```: Source languge used for training (i.e. zh in zh->vi or ja in ja->ko)
* ```target_lang```: Target language used for evaluation/ testing (i.e. vi in zh->vi or ko in ja->ko)


## Citation
If you find our ideas, code or dataset helpful, please consider citing our work as follows:
<pre>
@inproceedings{nguyen-etal-2023-enhancing,
    title = "Enhancing Cross-lingual Transfer via Phonemic Transcription Integration",
    author = "Nguyen, Hoang  and
      Zhang, Chenwei  and
      Zhang, Tao  and
      Rohrbaugh, Eugene  and
      Yu, Philip",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.583",
    pages = "9163--9175",
}
</pre>

## Acknowledgement
Our code is adapted from [XTREME Repository](https://github.com/google-research/xtreme) but preserves most of the training and evaluation procedures. </br>
