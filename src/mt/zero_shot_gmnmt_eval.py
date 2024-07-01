import os
import argparse
import json
import pandas as pd
import evaluate
import torch
from comet import download_model, load_from_checkpoint
from google.cloud import translate
from tqdm import tqdm


NO_OF_GPUs = torch.cuda.device_count()


def request_gmnmt(text, project_id):
    client = translate.TranslationServiceClient()
    
    location = "global"
    model_id = "general/nmt"    # requesting NMT model
    parent = f"projects/{project_id}/locations/{location}"
    model_path = f"{parent}/models/{model_id}"

    response = client.translate_text(
        request={
            "parent": parent,
            "contents": text,
            "model": model_path,
            "mime_type": "text/plain",  
            "source_language_code": "yo",
            "target_language_code": "en-US",
        }
    )
    
    return response


# sends batched requests to GMNMT
def get_translations(text, project_id, batch_size=10):
    translations = []
    for i in tqdm(range(0, len(text), batch_size)):
        j = i + batch_size if i + batch_size < len(text) else None
        batch = text[i:j]

        response = request_gmnmt(batch, project_id=project_id)
        translations += [x.translated_text for x in response.translations]
    
    return translations


def do_evaluate(source_column, data_df):
    # Sentence level metrics
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
    parser = argparse.ArgumentParser(description="Translate text with Google NMT.")
    parser.add_argument("--project_id", type=str, required=True, help="Project ID on Google Cloud (note that GOOGLE_API_KEY environment variable must be set).")
    parser.add_argument("--test_set", type=str, required=True, help="The source text to evaluate.")
    parser.add_argument("--cache_dir", type=str, required=True, help="Cache_directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save files.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    os.environ["GOOGLE_CLOUD_PROJECT"] = args.project_id

    # Load metrics
    comet_model_path = download_model("masakhane/africomet-mtl", saving_directory=args.cache_dir)
    comet_model = load_from_checkpoint(comet_model_path)
    chrf = evaluate.load("chrf", cache_dir=args.cache_dir)
    sacrebleu = evaluate.load("sacrebleu", cache_dir=args.cache_dir)

    test_data = pd.read_csv(args.test_set)
    
    metric_dict = {}
    for column in ['std_text', 'ife_text', 'ilaje_text', 'ijebu_text']:
        dialect_name = "_".join(column.split(" "))

        test_data[f"{dialect_name} translation"] = get_translations(test_data[dialect_name], args.project_id)

        print(f"Finished translating {dialect_name}")
        print(f"Computing metrics for {dialect_name}")

        # Compute BlEU / COMET/ CHRF and save metrics
        _, bleu_corpus, chrf_corpus, comet_corpus = do_evaluate(source_column=column, data_df=test_data)

        # Corpus bleu / chrf / comet
        metric_dict[f"{dialect_name}_bleu"] = bleu_corpus
        metric_dict[f"{dialect_name}_chrf"] = chrf_corpus
        metric_dict[f"{dialect_name}_comet"] = comet_corpus

        print(f"Finished with {column}")

    # Save new df and save metrics
    model_name = "gmnmt"
    test_data.to_csv(f"{args.output_dir}/{model_name}_translations.csv", index=False)

    # Store the corpus-level scores in a JSON file
    output_file = os.path.join(args.output_dir, f'{model_name}_scores.json')
    with open(output_file, 'w') as json_file:
        json.dump(metric_dict, json_file, indent=2)

