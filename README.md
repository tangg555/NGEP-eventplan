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
       ├── `roc-stories`        # a publicly availabl dataset: ROCStories
       ├── `verb-roc-stories`        # make it for testing but not used in the paper.
   └── thu-coai-hint		# HINT model needs it
├── preprocessing      # the code about automatical event extraction and event planning
├── resources      # resources for raw data, vanilla pretrained checkpoint, and so on.
├── src      # all the source code related to the models is put here
   └── configuration	# read the name, and you can understand what it is for.
   └── models	
   └── modules	
   └── utiles	
├── tasks      # the code to control experiments
   └── event-plan 	# our special way (the proposed method) to do event planning
   └── generation_models 	# for most of the experiments of event planning and story generation
```
If there is something you could not understand, please try to find the explanation in [my paper](https://arxiv.org/pdf/2210.10602.pdf).

If you are a freshman in NLG area and feel hard to read the code, I prepared a story generation demo for you ([demo](https://github.com/tangg555/story-generation-demo)). 
Because I believe the best code should be readable even without the code comments, I usually tried my best to design the data structure instead of writings code comments.

## Prerequisites
If you want to run this code, you must have at least meet the following requirement:
- Python 3 or Anaconda (mine is v3.8)
- [Pytorch](https://pytorch.org/) (mine is v1.11.0)
- transformers (a package for [huggingface](https://huggingface.co/facebook/bart-base)) v4.19.4
- [pytorch-lightning (a package)](https://www.pytorchlightning.ai/) v1.6.0
- all the packages listed in the file `requirements.txt` 

## Quick Start

#### 1. Install packages
```shell
python -r requirements.txt
```

#### 2. Collect Datasets and Resources
`datasets` and `resources` are separate from the code, since they are too large. 
Both of them can be downloaded from [BaiduNetDisk](https://pan.baidu.com/s/1gLxOZI0t65l4a6cTns8U2w) (input code: gb1a) or [Dropbox](https://www.dropbox.com/s/p9a4lz0eqax55it/datasets_and_resources.zip?dl=0). 
Put them to the basedir after downloaded.

If you intend to preprocess the data by yourself, please read following instructions. Otherwise, please skip to the next section.
#####2.1 Datasets
The **raw dataset** of roc story can be accessed for free. Google and get it. e.g. [homepage](https://cs.rochester.edu/nlp/rocstories/) .

train, val, test are split by the ratio of 0.90, 0.05, 0.05

the example of `test.source.txt` (leading context):

`ken was driving around in the snow .`

the example of `test.target.txt` (story):

`he needed to get home from work . he was driving slowly to avoid accidents . unfortunately the roads were too slick and ken lost control . his tires lost traction and he hit a tree . `
#####2.1 Resources
The structure of resources should be like this:
```markdown
├── resources
   └── external-models		# put vanilla pretrained checkpoint
   └── raw_data		# for preprocessing
```
The huggingface pretrained models (e.g. `bart-base`) can be downloaded from [here](https://huggingface.co/facebook/bart-base). Or you can directly set `--model_name_or_path=facebook/bart-base`, the code will download it for you.

#### 3. Run the code for training or testing
For training, the code entry is
`./tasks/xxx/train.py`; For testing, the code entry is `./tasks/xxx/test.py`

**Or** 

If you want to modify parameters, you can run
```shell
python tasks/story-generation/train.py --data_dir=datasets/story-generation/roc-stories\
 --learning_rate=5e-5 \
 --train_batch_size=16 \
 --eval_batch_size=10 \
 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/story-generation \
 --model_name leading-bart \
 --experiment_name=leading-bart-roc-stories\
 --val_check_interval=1.0 \
 --limit_val_batches=10 \
 --max_epochs=3 \
 --accum_batches_args=4
```

## Notation
Some notes for this project.
#### 1 - Additional directories and files in this project
```markdown
├── output  # this will be automatically created to put all the output stuff including checkpoints and generated text
├── .gitignore # used by git
├── requirement.txt # the checklist of essential python packages 
```
#### 2 - Scripts for Downloading huggingface models
I wrote two scripts to download models from huggingface website.
One is `tasks/download_hf_models.sh`, and another is `src/utils/huggingface_helper.py`

## Citation
If you found this repository or paper is helpful for you, please cite our paper. It is accepted by AACL 2022, but currently the citations of AACL papers have not come out yet.

This is the arxiv citation:
```angular2
@article{tang2022ngep,
  title={NGEP: A Graph-based Event Planning Framework for Story Generation},
  author={Tang, Chen and Zhang, Zhihao and Loakman, Tyler and Lin, Chenghua and Guerin, Frank},
  journal={arXiv preprint arXiv:2210.10602},
  year={2022}
}
```

