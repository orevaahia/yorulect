import os
import pandas as pd
import numpy as np
import evaluate
import argparse
from datasets import Dataset
from comet import download_model, load_from_checkpoint
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
import warnings 
warnings.filterwarnings('ignore')


def preprocess_data(df, tokenizer):
    def preprocess_func(row):
        return tokenizer(row["src_txt"], text_target=row["tgt_txt"], truncation=True)
    
    replace_punct = {
        '“': '"',
        '”': '"',
        '’': "'",
        '‘': "'",
        '–': '-'
    }
    df = df.replace(replace_punct, regex=True)

    dataset = Dataset.from_pandas(df)
    encoded_dataset = dataset.map(preprocess_func, batched=True, batch_size=16)  
    encoded_dataset = encoded_dataset.select_columns(['input_ids', 'attention_mask', 'labels'])    # training doesn't work if there are text columns
    
    return encoded_dataset.with_format("torch")


# converts df from one dialect per column,
# to one data point per row, for each dialect we're training with
def transform_df(df, single_dialect=None, reverse_direction=False):

    dialects = ["Ife Dialect", 
                "Ilaje Dialect", 
                "Ijebu Dialect", 
                "Standard Yoruba"] if not single_dialect else [f"{single_dialect} Dialect"]
    dialects = ["Standard Yoruba"] if single_dialect == "Standard" else dialects

    joint_df = pd.DataFrame()
    for dialect in dialects:
        dialect_df = df[[dialect, "English"]]
        dialect_df["dialect"] = dialect
        if reverse_direction:
            dialect_df = dialect_df.set_axis(["tgt_txt", "src_txt", "dialect"], axis=1)
        else:
            dialect_df = dialect_df.set_axis(["src_txt", "tgt_txt", "dialect"], axis=1)
        joint_df = pd.concat([joint_df, dialect_df])
    
    return joint_df


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def get_comet_score(src_text, mt_text, ref_text):
    data = []
    for src_s, mt_s, ref_s in zip(src_text, mt_text, ref_text):
        data.append({"src": " ".join(src_s),
                    "mt": " ".join(mt_s),
                    "ref": " ".join(ref_s)})

    model_output = comet_model.predict(data, batch_size=4)

    # scores[0] are all scores, scores[1] is average, scores[2] is metadata
    scores = model_output.to_tuple()
    return scores[1]


def postprocess_text(preds, labels, inputs):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    inputs = [input.strip() for input in inputs]

    return preds, labels, inputs


def compute_metrics(eval_preds):
    preds, labels, inputs = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # -100 is inserted by the trainer as the pad token, but not properly decoded
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
    decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)

    decoded_preds, decoded_labels, decoded_inputs = postprocess_text(decoded_preds, decoded_labels, decoded_inputs)

    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {
        "bleu": bleu_result["score"],
        "comet": get_comet_score(decoded_inputs, decoded_preds, decoded_labels)
    }

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def load_model(model_checkpoint, src_lang="yor_Latn", tgt_lang="eng_Latn", add_lang_codes=None, cache_dir=None):
    tokenizer = NllbTokenizer.from_pretrained(model_checkpoint, 
        additional_special_tokens=add_lang_codes, 
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        cache_dir=cache_dir
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, cache_dir=cache_dir)

    if add_lang_codes:
        # resize embedding layer to account for new tokens
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
        
        # initialize new lang tokens with yoruba embedding
        yo_idx = tokenizer.added_tokens_encoder["yor_Latn"]
        yo_embed = model.model.shared.weight.data[yo_idx]
        for new_lang in add_lang_codes:
            idx = tokenizer.added_tokens_encoder[new_lang]
            model.model.shared.weight.data[idx] = yo_embed

    return model, tokenizer



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finetune NLLB on Yoruba data.')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset, including train and validation csvs.')
    parser.add_argument('--model_output_dir', type=str, required=True, help='Where to store model artifacts.')
    parser.add_argument('--menyo_dir', type=str, help='Path to Menyo data. If supplied, training will be done jointly with MENYO-20k data.')
    parser.add_argument("--cache_dir", type=str, help="Cache_directory.")
    parser.add_argument('--reverse_direction', action='store_true', help='To train en2yo(default is yo2en).')
    parser.add_argument('--epochs', default=4, type=int, help='Number of training epochs.')
    parser.add_argument('--batch_size', default=16, type=int, help='Desired effective batch size.')
    parser.add_argument('--num_gpus', default=1, type=int, help='How many GPUs to use for training.')
    parser.add_argument('--lr', default=2e-5, type=float, help='Learning rate.')
    parser.add_argument('--ga', default=1, type=int, help='Gradient accumulation steps.')
    parser.add_argument('--trained_model_name', help='Name of trained model.')
    parser.add_argument('--checkpoint', help='Checkpoint to use for training (default facebook nllb-200-distilled-600M).')
    parser.add_argument('--single_dialect', type=str, help='Which dialect to train on (default behavior is to train all dialects jointly).')
    parser.add_argument('--wandb', action='store_true', help='Whether to use wandb experiment tracking (must have set WANDB_API_KEY environment variable).')
    parser.add_argument('--push_to_hub', action='store_true', help='Whether to push trained model to HuggingFace Hub (must have set HF_TOKEN environment variable).')
    args = parser.parse_args()

    model_checkpoint = args.checkpoint if args.checkpoint else "facebook/nllb-200-distilled-600M"

    if args.wandb:
        os.environ["WANDB_PROJECT"] = "yoruba-dialects" 

    bleu_metric = evaluate.load("sacrebleu")
    model_path = download_model("masakhane/africomet-mtl")
    comet_model = load_from_checkpoint(model_path)

    # add new dialect if training dialect
    dialect_to_lang_code = {
        "Ife": "ife_Yoru", 
        "Ilaje": "ila_Yoru", 
        "Ijebu": "ije_Yoru", 
    }
    new_dialect = [dialect_to_lang_code[args.single_dialect]] if args.single_dialect in dialect_to_lang_code else None
    
    # set src and tgt langs
    yor_lang = new_dialect[0] if new_dialect else "yor_Latn"
    src_lang = "eng_Latn" if args.reverse_direction else yor_lang
    tgt_lang = yor_lang if args.reverse_direction else "eng_Latn"

    print(f"{new_dialect=} {src_lang=} {tgt_lang=}")
    
    # load model, create new lang codes if applicable
    model, tokenizer = load_model(model_checkpoint, 
                                  src_lang=src_lang, 
                                  tgt_lang=tgt_lang, 
                                  add_lang_codes=new_dialect,
                                  cache_dir=args.cache_dir)

    # load and process data
    raw_train_df = pd.read_csv(f"{args.dataset_dir}/train.csv")
    train_df = transform_df(raw_train_df, single_dialect=args.single_dialect, reverse_direction=args.reverse_direction)
    
    # concatenate menyo data, if using 
    if args.menyo_dir:
        menyo_df = pd.read_csv(f"{args.menyo_dir}/train.tsv", sep='\t')
        if args.reverse_direction:
            menyo_df = menyo_df.set_axis(["src_txt", "tgt_txt"], axis=1)
        else:
            menyo_df = menyo_df.set_axis(["tgt_txt", "src_txt"], axis=1)
        train_df = pd.concat([train_df, menyo_df])
    
    print(len(train_df), " training samples")
    train_ds = preprocess_data(train_df, tokenizer)

    # load eval data
    raw_dev_df = pd.read_csv(f"{args.dataset_dir}/validation.csv")
    eval_df = transform_df(raw_dev_df, single_dialect=args.single_dialect, reverse_direction=args.reverse_direction)
    eval_ds = preprocess_data(eval_df, tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_checkpoint)

    no_gpus = args.num_gpus
    print(f"Training on {no_gpus=}, {args.ga=}, {args.batch_size=}")

    model_name = args.trained_model_name if args.trained_model_name else f"yo_nllb_lr{args.lr}_b{args.batch_size}"
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{args.model_output_dir}/{model_name}",
        run_name=model_name,
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        seed=0,
        per_device_train_batch_size= args.batch_size // no_gpus // args.ga,
        per_device_eval_batch_size= args.batch_size // no_gpus // args.ga, 
        gradient_accumulation_steps=args.ga,
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy = "epoch",
        push_to_hub=args.push_to_hub,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=1,    
        logging_strategy="epoch",  
        include_inputs_for_metrics=True,  
        log_level="error",
        report_to=("wandb" if args.wandb else None),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

        

