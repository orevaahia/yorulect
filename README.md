# WIP !
# Voices Unheard: NLP Resources and Models for Yorùbá Regional Dialects

This repository contains the code for the paper, Link to paper - []().


## 1. Data
Download the data folder from this [drive](https://drive.google.com/file/d/1WCFqRzcmpXxtzUlRAAnGaqOV079JFccU/view?usp=sharing).


### Data Description
This dataset contains corpus of high quality, contemporary Yorùbá speech and text data parallel across four Yorùbá dialects; Standard Yorùbá, Ifè, Ìlàje and Ìjèbú in three domains (religious, news, and Ted talks). The dataset can be used in (text-to-text) machine translation (MT), automatic speech recognition (ASR), speech-to-text translation (S2TT), and speech-to-speech translation (STST) tasks.

## 2. Running the Code:

Clone the repository and install requirements

```
git clone https://github.com/orevaahia/yorulect
cd yorulect
pip install -r requirements.txt
```

## Machine Translation:

### Zero-shot:
```
# zero-shot evaluation of Aya and MT0
bash scripts/mt/zero_shot_lm.sh

# zero-shot evaluation of NLLB and M2M-100
bash scripts/mt/zero_shot_mt.sh

# zero-shot evaluation of Google Translate
bash scripts/mt/zero_shot_gmnmt.sh
```

### Finetuning [NLLB](https://huggingface.co/facebook/nllb-200-distilled-600M) :
```
bash scripts/mt/finetune_mt.sh
```

## Automatic Speech Recognition:
### Zero-shot:
```
# zero-shot evaluation of MMS and Whisper
bash scripts/speech/zero_shot_asr.sh
```
### Finetuning [MMS](https://huggingface.co/facebook/mms-1b-all) and [XLSR](https://huggingface.co/facebook/wav2vec2-xls-r-1b):
```
# MMS
bash scripts/speech/finetune_mms_asr.sh

# XSLR
bash scripts/speech/finetune_xslr_asr.sh
```

## Citation:
