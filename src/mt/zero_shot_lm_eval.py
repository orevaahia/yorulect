import evaluate
import argparse
import json
import os
import pandas as pd
import torch
import warnings

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from comet import download_model, load_from_checkpoint

NO_OF_GPUs = torch.cuda.device_count()
s

def translate(source_column, data_df, prompt, pipe_fn):
    data_df[f"{source_column} prompt"] = data_df[source_column].apply(lambda x: f"{prompt}: "+ x)
    data_df[f"{source_column} translation"] = data_df[f"{source_column} prompt"].apply(lambda x: pipe_fn(x, max_new_tokens=256)[0]["generated_text"])

    print(f"Finished translating {source_column}")
    print(f"Computing metrics for {source_column}")

    data_df[f"{source_column} bleu"] = data_df.apply(lambda x: sacrebleu.compute(predictions = [x[f"{source_column} translation"]], references=  [x["english_text"]])["score"], axis=1  )
    data_df[f"{source_column} chrf"] = data_df.apply(lambda x: chrf.compute(predictions = [x[f"{source_column} translation"]], references=  [x["english_text"]])["score"], axis=1  )

    # Compute comet metrics
    translations = data_df[f"{column} translation"].to_list()
    references = data_df["english_text"].to_list()
    sources = data_df[source_column].to_list()

    comet_dict = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(sources, translations, references)]
    comet_score = comet_model.predict(comet_dict, batch_size=8, gpus=NO_OF_GPUs)


    data_df[f"{source_column} comet"] = comet_score[0]

    # Compute Corpus Bleu and Chrf
    bleu_score = sacrebleu.compute(predictions=translations, references=references)["score"]
    chrf_score = chrf.compute(predictions=translations, references=references)["score"]

    return data_df, bleu_score, chrf_score, comet_score[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate text with LLMS.")
    parser.add_argument("--model_name", type=str, required=True, help="The HuggingFace model name.")
    parser.add_argument("--test_set", type=str, required=True, help="The source text to evaluate.")
    parser.add_argument("--cache_dir", type=str, required=True, help="Cache_directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save files .")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, cache_dir=args.cache_dir)

    # Load comet model and metric
    comet_model_path = download_model("masakhane/africomet-mtl", saving_directory=args.cache_dir)
    comet_model = load_from_checkpoint(comet_model_path)

    #pipe = pipeline("text2text-generation", model=args.model_name,  device_map="auto", model_kwargs={"cache_dir":args.cache_dir})
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device_map="auto")


    # Load Dataset (We only need the test set )
    test_data = pd.read_csv(args.test_set)

    # Load metrics
    chrf = evaluate.load("chrf", cache_dir=args.cache_dir)
    sacrebleu = evaluate.load("sacrebleu", cache_dir=args.cache_dir)

    metric_dict = {}
    # Translate
    #for column in ['Standard Yoruba', 'Ife Dialect', 'Ijebu Dialect', 'Ilaje Dialect']:
    for column in ['std_text', 'ife_text', 'ilaje_text', 'ijebu_text']:
        dialect_name = "_".join(column.split(" "))
        #translate(source_column=column, data_df=test_data, prompt="Translate to English: ")
        # Compute BlEU / COMET/ CHRF and save metrics
        _, bleu_corpus, chrf_corpus, comet_corpus = translate(source_column=column, data_df=test_data, prompt="Translate to English: ", pipe_fn=pipe)


        # Corpus bleu / chrf / comet
        metric_dict[f"{dialect_name}_bleu"] = bleu_corpus
        metric_dict[f"{dialect_name}_chrf"] = chrf_corpus
        metric_dict[f"{dialect_name}_comet"] = comet_corpus

        print(f"Finished with {column}")

    # Save new df and save metrics
    model_name = args.model_name.split("/")[1]
    test_data.to_csv(f"{args.output_dir}/{model_name}_translations.csv", index=False)

    # Store the corpus-level scores in a JSON file
    output_file = os.path.join(args.output_dir, f'{model_name}_scores.json')
    with open(output_file, 'w') as json_file:
        json.dump(metric_dict, json_file, indent=2)



