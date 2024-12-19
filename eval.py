import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, default_data_collator, \
    DataCollatorForLanguageModeling
from datasets import load_dataset

from tqdm import tqdm
import random
import math
import csv
import json


def load_model(save_directory='save_directory'):
    model = AutoModelForCausalLM.from_pretrained(save_directory, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(save_directory)
    print("Model and tokenizer loaded successfully!")
    return model, tokenizer


def save_csv_file(results, output_dir='./outputs/', filename='unlearned_models_output'):
    filename = output_dir + filename + '.csv'
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(results.keys())
        writer.writerows(zip(*results.values()))

    print(f"Outputs saved to {filename}")


def save_json_file(results, output_dir='./outputs/', filename='unlearned_models_output'):
    output_file = output_dir + filename + '.json'
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Outputs saved to {output_file}")


def create_batches(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        yield dataset.select(range(i, min(i + batch_size, len(dataset))))


def evaluate_model(model, tokenizer, dataset, output_dir='./outputs/', filename='unlearned_models_output',
                   language='en',
                   device="cuda:0"):
    model.to(device)
    batch_size = 32
    results = {"question": [], "target": [], "model_answer": []}
    total_batches = math.ceil(len(dataset) / batch_size)

    # Process the dataset in batches
    for batch in tqdm(create_batches(dataset, batch_size), desc="Processing batches", total=total_batches):
        batch_questions = []
        batch_targets = []
        batch_inputs = []

        for sample in batch:
            q = sample['MCQ_Question']
            original_options = [sample['Option_1'], sample['Option_2'], sample['Option_3'], sample['Option_4']]
            target_output = sample[sample['Correct_Option']]

            unknown_options_en = [
                "Cannot be determined",
                "Unknown",
                "Not enough information",
                "Unclear",
            ]

            unknown_options_fr = [
                "Impossible à déterminer",
                "Inconnu",
                "Pas assez d'informations",
                "Peu clair",
            ]

            unknown_options_hi = [
                "निर्धारित नहीं किया जा सकता",
                "अज्ञात",
                "पर्याप्त जानकारी नहीं है",
                "अस्पष्ट",
            ]

            if language == 'en':
                unknown_option = random.choice(unknown_options_en)
            elif language == 'hi':
                unknown_option = random.choice(unknown_options_hi)
            else:
                unknown_option = random.choice(unknown_options_fr)

            all_options = original_options + [unknown_option]
            random.shuffle(all_options)
            labeled_options = [f"{label}){option}" for label, option in zip(["a", "b", "c", "d", "e"], all_options)]
            if language == 'en':
                inp_seq = f"Question: {q}\n" + "\n".join(labeled_options) + "\nChoose one of the options above."
            elif language == 'hi':
                inp_seq = f"सवाल: {q}\n" + "\n".join(labeled_options) + "\nउपरोक्त विकल्पों में से एक चुनें."
            else:
                inp_seq = f"Question : {q}\n" + "\n".join(labeled_options) + "\nChoisissez une des options ci-dessus."

            batch_questions.append(inp_seq)
            batch_targets.append(target_output)
            batch_inputs.append(inp_seq)

        tokenized_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(**tokenized_inputs, max_length=256, do_sample=False)

        # Collect results for the batch
        for i in range(len(batch_questions)):
            start_index = len(tokenized_inputs['input_ids'][i])
            results["question"].append(batch_questions[i])
            results["target"].append(batch_targets[i])
            results["model_answer"].append(tokenizer.decode(outputs[i][start_index:]))

    save_csv_file(results, output_dir=output_dir, filename=filename)
    save_json_file(results, output_dir=output_dir, filename=filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation of Unlearned Models")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory of saved models")
    parser.add_argument("--dataset", type=str, required=True, help="Directory of saved models")
    parser.add_argument("--output_dir", type=str, default="./outputs/", help="Directory to save evaluation outputs")
    parser.add_argument("--filename", type=str, default="unlearned_models_output_fr", help="Output file name")
    parser.add_argument("--language", type=str, choices=["en", "hi", "fr"], default="en",
                        help="Language for evaluation (options: en, hi, fr)")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_dir)
    dataset = load_dataset("csv", data_files=args.dataset)
    dataset = dataset['train']
    model.to("cuda:0")
    evaluate_model(
        model,
        tokenizer,
        dataset,
        output_dir=args.output_dir,
        filename=args.filename,
        language=args.language)
