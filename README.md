# Exploring the Importance of Source Text in Automatic Post-Editing for Context-Aware Machine Translation

The code is modified based on [Nematus](https://github.com/EdinburghNLP/nematus). Three branch are there in the respository, **master**: implement Transference and adapt to the document-level scenorio; **2-way**: the mirror implementation of *master* branch with single input; **scripts**: scripts and configuration files to train and evaluate models.

[paper](https://www.aclweb.org/anthology/2021.nodalida-main.34/), [poster](https://github.com/zippotju/Context-Aware-Bilingual-Repair-for-Neural-Machine-Translation/blob/master/source/poster.pdf)

## 1. Requirements

Python 3.6

Tensorflow 1.12

## 2. Download data and Preprocessing

Download the data from [here](https://github.com/lena-voita/good-translation-wrong-in-context/#training-data). Preprocess data following steps described [here](https://github.com/lena-voita/good-translation-wrong-in-context/#data-preprocessing).

## 3. Model Training and Evaluation

- Download codebases for Transference and DocRepair separately:

```
git clone https://github.com/zippotju/Context-Aware-Bilingual-Repair-for-Neural-Machine-Translation.git
git clone https://github.com/zippotju/Context-Aware-Bilingual-Repair-for-Neural-Machine-Translation.git --branch 2-way --single-branch Context-Aware-Bilingual-Repair-for-Neural-Machine-Translation-2-way
```

- Download scripts for training and evaluation

```
git clone https://github.com/zippotju/Context-Aware-Bilingual-Repair-for-Neural-Machine-Translation.git --branch scripts --single-branch scripts
```

- Download consistency test sets

```
git clone https://github.com/lena-voita/good-translation-wrong-in-context.git
```

- Following instructions in *scripts* branch to train and evaluate the models.

## 4. Instruction
*respository-specific usage instruction, for general usages instruction, please refer to [Nematus](https://github.com/EdinburghNLP/nematus)*

### `nematus/train.py` : use to train a new model

### data sets; model loading and saving
| parameter | description |
|---        |---          |
| --mt_dataset PATH | path to synthetic training corpus (mt) |
| --f_source_dataset PATH | path to ParData (src) |
| --f_mt_dataset PATH | path to ParData (mt) |
| --f_target_dataset PATH | path to ParData (ref) |
| --data_mode | Format of the training data. 'single': context agnostic sentence-level parallel data; 'multiple': context-aware four consistent sentences grouped in a sample (default: single) |
| --beam_mode_pro | probability for selecting beam-searched round-trip translations (mt) as input (default: 0) |
| --noise_mode_pro | probability for selecting noisy synthetic translations (mt) as input (default: 0) |
| --sample_mode_pro | probability for selecting randomly-sampled round-trip translations (mt) as input (default: 1) |
| --beam_dropout | word dropout probability of beam-searched round-trip translations (default: 0.1) |
| --noise_dropout | word dropout probability of ground-truth translations to produce noisy synthetic translations  (default: 0.2) |
| --sample_dropout | word dropout probability of randomly-sampled round-trip translations (default: 0.1) |
| --noise_source | noise the source-side training data by: (1) deleting a token; (2) replacing a token with a rondom token; (3) swapping two tokens, with the probability of hyperparameters-`beam_dropout`. (default: False)|
| --f_ratio | the percentage of ParData in training data (default: 0) |

### training parameters
| parameter | description |
|---        |---          |
| --weighted_conservative_loss | weighting loss to enforces the system to be more conservative and learn fewer edits (default: False) |

### validation parameters
| parameter | description |
|---        |---          |
| --valid_mt_dataset PATH | path to mt validation corpus (default: None) |
| --valid_consis_script PATH | path to script for external consistency test validation (default: None) |

### translate parameters
| parameter | description |
|---        |---          |
| --conservative_penalty FLOAT | the conservative penalty parameters (c) (default: 0) |
| --conservative_way {probabilities, logits} | the place where conservative penalty assigns, either on probabilities or logits (default: probabilities) |
| --add_eos | see <EOS> tokens as a existing token in the source/mt sentences and do not give punishment when generate it (default: False) |


### `nematus/translate.py` : use an existing model to translate a source (and mt) text

| parameter | description |
|---        |---          |
| -i PATH, --input PATH | input file (default: standard input) |
| -ip PATH, --input_p' PATH | input file of mt (default: standard input) |
| -o PATH, --output PATH | output file (default: standard output) |
| -cp FLOAT, --conservative_penalty FLOAT | the conservative penalty parameters (c) (default: 0) |
| -cw, --conservative_way {probabilities, logits} | the place where conservative penalty assigns, either on probabilities or logits (default: probabilities) |
| -ae, --add_eos | see <EOS> tokens as a existing token in the source/mt sentences and do not give punishment when generate it (default: False) |


### `nematus/score.py` : use an existing model to score a parallel corpus

| parameter | description |
|---        |---          |
| -mt PATH, --mt PATH | mt text file |
