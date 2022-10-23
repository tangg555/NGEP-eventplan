# NGEP-eventplan
This repository is the code and other resources for the paper [NGEP: A Graph-based Event Planning Framework for Story Generation](https://arxiv.org/pdf/2210.10602.pdf) 

To make sure everyone can easily reproduce our work, I will do my best to ensure every essential resource is included in this repository, and the README covers all the information you need to implement our work.

## Introduction
This project is implemented with **Pytorch**.

This project is implemented based on [pytorch-lightning](https://www.pytorchlightning.ai/) framework, a framework to ease the training of pytorch. If you dislike it, no worry, you can copy the model files (in `src/models`) and datasets files (in `src/modules`) to your own repository, and train them with your code.

All the pretrained model used here are downloaded from [Huggingface](https://huggingface.co/docs). E.g. [BART](https://aclanthology.org/2020.acl-main.703.pdf) is downloaded from [Hugginface: bart-base](https://huggingface.co/facebook/bart-base).

The code is organized as follows:
```markdown
├── datasets
   └── event-plan		# expeirment group name
       ├── `roc-stories`        # a publicly available dataset: ROCStories
   └── thu-coai-hint		# Testing HINT model will need it
├── preprocessing      # the code about automatical event extraction and event planning
├── resources      # resources for raw data, vanilla pretrained checkpoint, and so on.
├── src      # all the source code related to the models is put here
   └── configuration	# read the name, and you will know what it is for.
   └── models	
   └── modules	
   └── utils	
├── tasks      # the code to control experiments
   └── event-plan 	# our proposed method to do event planning
   └── generation_models 	# for most of the experiments of event planning and story generation
```
If there is something you could not understand, please try to find the explanation in [my paper](https://arxiv.org/pdf/2210.10602.pdf).

If you are a freshman in NLG area and feel hard to read the code, I prepared a story generation demo for you ([demo](https://github.com/tangg555/story-generation-demo)). 
I usually tried my best to design the data structure instead of writings code comments, because I believe a good code should be readable even without the code comments.

## Prerequisites
If you want to run this code, you have to at least satisfy the following requirement:
- Python 3 or Anaconda (mine is v3.8)
- [Pytorch](https://pytorch.org/) (mine is v1.11.0)
- transformers (a package for [huggingface](https://huggingface.co/facebook/bart-base)) v4.19.4
- [pytorch-lightning (a package)](https://www.pytorchlightning.ai/) v1.6.0
- all the packages listed in the file `requirements.txt` 

## Quick Start

### 1. Install packages
Install the aforementioned prerequisites, and run
```shell
python -r requirements.txt
```

### 2. Collect Datasets and Resources

`datasets` and `resources` are not included in the code, since their sizes are too large. 
Both of them can be downloaded from [Dropbox](https://www.dropbox.com/s/3uh7oylu9joqw9i/datasets_and_resources.zip?dl=0). 
Unzip it at the base directory.

If you intend to preprocess the data by yourself, please read following instructions. Otherwise, please skip to the next section.

#### 2.1 Datasets
**It is worth mentioning that:** In the given dataset, we also include the planned eventplan from **Neural Advisor**
 and **NGEP**. Their names are "xxx_bart_event.xxx.txt" -- **Neural Advisor**; "xxx_predicted_event.xxx.txt" -- **NGEP**.
E.g., "test_bart_event.source.txt" means the event plan of **Neural Advisor** for the test dataset.

**Preprocess**
Put your downloaded raw dataset (we downloaded it from [HINT](https://github.com/thu-coai/HINT)) to `resources/raw_data`, so that you will have `resources/raw_data/thu-coai-hint/roc-stories`.

Run `preprocessing/hint_roc_stories_helper.py`, and then `preprocessing/event_annotator.py`, and you will have `resources/datasets/event-plan/roc-stories`.

Similarly, if you want to run HINT as a story generation model for experiments, you need to download HINT dataset from [HINT](https://github.com/thu-coai/HINT), and make it to be `/datasets/thu-coai-hint/roc-stories`.

#### 2.2 Resources

The structure of resources should be like this:
```markdown
├── resources
   └── external-models		# put vanilla pretrained checkpoint
   └── raw_data		# for preprocessing
```
The huggingface pretrained models (e.g. `bart-base`) can be downloaded from [here](https://huggingface.co/facebook/bart-base). Or you can directly set `--model_name_or_path=facebook/bart-base`, the code will download it for you.

### 3. Run the code for training or testing

#### 3.1 Introduction

Experiments include two parts: (1) event planning aims to input **leading context** and output **event plan**;
(2) story generation aims to input **leading+eventplan** and out **stories**.

The project is big, so please read the codes in `tasks` to understand how it works.

**If you don't care the baselines and experiments**, please only read following files:
- (1) `tasks/event-plan/train.py` to get the model for **Neural Advisor** (I named it `event-plan-bart`).
- (2) `tasks/event-plan/predict.py` to use **Neural Advisor** and event graph for event planning (**NGEP**).

In case you don't want to train **Neural Advisor** by yourself, a checkpoint ([Dropbox](https://www.dropbox.com/s/l8duhtvlwzd6nz7/event-plan-bart-roc-stories.tar.gz?dl=0)) is released for your convenience. 
Put it somewhere and restore it with a command. (referring to `eventplan_commands.sh`)

#### 3.2  commands for NGEP

The user parameters settings are located in `src/configuration/event_plan/config_args.py` and 
`src/configuration/generation_models/config_args.py`.

For training **Neural Advisor**:
```shell
python tasks/event-plan/train.py --data_dir=datasets/event-plan/roc-stories\
 --learning_rate=1e-4 --train_batch_size=256 --eval_batch_size=10 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/event-plan --model_name event-plan-bart --experiment_name=event-plan-bart-roc-stories\
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=1 --num_sanity_val_steps=0
```

For event plan with **NGEP**:
```shell
python tasks/event-plan/predict.py
```

#### 3.3 All of the commands

We conduct a range of experiments to validate the effectiveness of our model, 
so it has plenty of commands. Please refer to the file `eventplan_commands.sh` 
to select the command you want to execute.

## Notation
Some notes for this project.
### 1 - Additional directories and files in this project
```markdown
├── output  # this will be automatically created to put all the output stuff including checkpoints and generated text
├── .gitignore # used by git
├── requirement.txt # the checklist of essential python packages 
```
### 2 - Scripts for Downloading huggingface models
I wrote two scripts to download models from huggingface website.
One is `tasks/download_hf_models.sh`, and another is `src/utils/huggingface_helper.py`

## Citation
If you found this repository or paper is helpful to you, please cite our paper. It is accepted by AACL 2022, but currently the citations of AACL papers have not come out yet.

This is the arxiv citation:
```angular2
@article{tang2022ngep,
  title={NGEP: A Graph-based Event Planning Framework for Story Generation},
  author={Tang, Chen and Zhang, Zhihao and Loakman, Tyler and Lin, Chenghua and Guerin, Frank},
  journal={arXiv preprint arXiv:2210.10602},
  year={2022}
}
```

