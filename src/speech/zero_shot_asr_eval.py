import argparse
import evaluate
import json
import os
import pandas as pd
import torch

from tqdm import tqdm
from pathlib import Path
from typing import Optional, Tuple, Union

from datasets import load_dataset, Audio
from sacrebleu.metrics.base import Score, Signature
from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF

from transformers.models.whisper.english_normalizer import BasicTextNormalizer, EnglishTextNormalizer
from transformers import Wav2Vec2ForCTC, AutoProcessor, pipeline

whisper_norm = BasicTextNormalizer()

def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""


def normalise(batch):
    batch["norm_text"] = whisper_norm(batch["transcription"])
    return batch

def compute_metrics(references, predictions, norm_references, norm_predictions, wer_metric, cer_metric):
        wer = wer_metric.compute(references=references, predictions=predictions)
        wer = round(100 * wer, 2)
        cer = cer_metric.compute(references=references, predictions=predictions)
        cer = round(100 * cer, 2)
        norm_wer = wer_metric.compute(references=norm_references, predictions=norm_predictions)
        norm_wer = round(100 * norm_wer, 2)
        norm_cer = cer_metric.compute(references=norm_references, predictions=norm_predictions)
        norm_cer = round(100 * norm_cer, 2)

        return wer, norm_wer, cer, norm_cer


def main(args):
    os.makedirs(args.output_dir, exist_ok = True)
    model_id = args.hf_model
    basename = os.path.basename(args.output_dir)
    wer_metric = evaluate.load("wer", cache_dir=args.cache_dir, experiment_id=f"{basename}wer")
    cer_metric = evaluate.load("cer", cache_dir=args.cache_dir, experiment_id=f"{basename}cer")

    dataset = load_dataset("audiofolder",
            data_dir=args.dataset,
            cache_dir=args.cache_dir
        )["train"]

    text_column_name = "transcription"
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(normalise, num_proc=2)
    dataset = dataset.filter(is_target_text_in_range, input_columns=[text_column_name], num_proc=2)

    predictions, references, norm_predictions, norm_references = [], [], [], []
    if "mms" in args.model_name:
        print("Evaluating the MMS model ")
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=args.cache_dir)
        processor.tokenizer.set_target_lang(args.language)

        model = Wav2Vec2ForCTC.from_pretrained(model_id, cache_dir=args.cache_dir).to("cuda")
        model.load_adapter(args.language)
        for input_speech in tqdm(dataset, desc='Decode Progress'):
            audio_features = input_speech["audio"]
            input_features = processor(audio_features["array"], sampling_rate=audio_features["sampling_rate"], return_tensors="pt").to("cuda")

            with torch.no_grad():
                try:
                    outputs = model(**input_features).logits
                except RuntimeError as e:
                    if "Calculated padded input size per channel: (0). Kernel size: (10). Kernel size can't be greater than actual input size" in str(e):
                        print("RuntimeError: Calculated padded input size per channel: (0). Kernel size: (10). Kernel size can't be greater than actual input size")
                        continue
                    else:
                        print(e)
                        continue
                predicted_ids = torch.argmax(outputs, dim=-1)[0]
                transcription = processor.decode(predicted_ids)

            reference = input_speech["transcription"]
            norm_reference = input_speech["norm_text"]
            transcription = processor.decode(predicted_ids, skip_special_tokens=True)

            predictions.append(transcription)
            references.append(reference)
            norm_predictions.append(whisper_norm(transcription))
            norm_references.append(norm_reference)

    elif "whisper" in args.model_name:
        print("Evaluating the whisper model ")

        def data(dataset):
            for i, item in enumerate(dataset):
                yield {**item["audio"], "reference": item["transcription"], "norm_reference": item["norm_text"]}

        whisper_asr = pipeline(
        "automatic-speech-recognition", model=model_id, device=args.device,
        model_kwargs={"cache_dir": args.cache_dir})

        whisper_asr.model.config.forced_decoder_ids = (
            whisper_asr.tokenizer.get_decoder_prompt_ids(
                language=args.language, task="transcribe"
            )
        )
        for out in tqdm(whisper_asr(data(dataset), batch_size=args.batch_size), desc='Decode Progress'):
            predictions.append(out["text"])
            references.append(out["reference"][0])
            norm_predictions.append(whisper_norm(out["text"]))
            norm_references.append(out["norm_reference"][0])


    wer, norm_wer, cer, norm_cer= compute_metrics(references, predictions, norm_references, norm_predictions, wer_metric, cer_metric)
    print("\nWER : ", wer)
    print("CER : ", cer)
    print("\nNORMALIZED WER : ", norm_wer)
    print("NORMALIZED CER : ", norm_cer)

    result_file = open(os.path.join(args.output_dir, "predictions.txt"), 'w')
    result_file.write('\nWER: ' + str(wer) + '\n')
    result_file.write('CER: ' + str(cer) + '\n')
    result_file.write('\nNORMALIZED WER: ' + str(norm_wer) + '\n')
    result_file.write('NORMALIZED CER: ' + str(norm_cer) + '\n\n\n')

    for ref, hyp in zip(references, predictions):
        result_file.write('REF: ' + ref + '\n')
        result_file.write('HYP: ' + hyp + '\n')
        result_file.write("------------------------------------------------------" + '\n')
    result_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="openai/whisper-tiny",
        help="Huggingface model name. Example: openai/whisper-tiny",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        default="cache",
        help="Directory to cache models and datasets",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=False,
        default="yo",
        help="Two letter language code for the transcription language, e.g. use 'hi' for Hindi. This helps initialize the tokenizer.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=16,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="predictions_dir",
        help="Output directory for the predictions and hypotheses generated.",
    )
    args = parser.parse_args()
    main(args)