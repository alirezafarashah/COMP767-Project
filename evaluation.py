import os
import re
import json
import argparse
from tqdm import tqdm

def convert_to_jsonl(input_file, output_file):

    with open(input_file, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    questions = data["question"]
    targets = data["target"]
    model_answers = data["model_answer"]

    if len(questions) != len(targets) or len(targets) != len(model_answers):
        raise ValueError("The lists in the input JSON are not of the same length!")

    with open(output_file, "w", encoding="utf-8") as outfile:
        for question, target, model_answer in zip(questions, targets, model_answers):
            line = {
                "question": question,
                "target": target,
                "model_answer": model_answer
            }
            json.dump(line, outfile, ensure_ascii=False)  
            outfile.write("\n")

    print(f"Converted {input_file} to JSONL and saved as {output_file}")


def extract_answer_regex(question, model_answer):

    answer_match = re.search(r"(उत्तर:|Answer:|Réponse:)\s*[a-eA-E]\)\s*([^\n]+)", model_answer, re.UNICODE)
    if answer_match:
        return answer_match.group(2).strip()

    options_pattern = r"\b[a-eA-E]\)\s*([^\n]+)"
    options = re.findall(options_pattern, question)
    if not options:
        return "No options found in question"

    processed_answer = ' '.join(model_answer.split())

    for option in options:
        option_clean = option.strip()
        pattern = re.compile(r'\b' + re.escape(option_clean) + r'\b', re.IGNORECASE)
        if pattern.search(processed_answer):
            return option_clean

    return "Unknown"


def evaluate_file(input_file, output_file, language):

    unknown_options = {
        "en": ["Cannot be determined", "Unknown", "Not enough information", "Unclear"],
        "fr": ["Impossible à déterminer", "Inconnu", "Pas assez d'informations", "Peu clair"],
        "hi": ["निर्धारित नहीं किया जा सकता", "अज्ञात", "पर्याप्त जानकारी नहीं है", "अस्पष्ट"]
    }

    unknown_set = set(unknown_options[language])

    target_matches = 0
    unknown_matches = 0
    other_matches = 0
    total_samples = 0

    with open(input_file, 'r', encoding="utf-8") as infile, open(output_file, 'w', encoding="utf-8") as outfile:
        for line in tqdm(infile, desc="Processing"):
            total_samples += 1
            data = json.loads(line)
            question = data.get("question", "")
            target = data.get("target", "")
            model_answer = data.get("model_answer", "")
            
            extracted_answer = extract_answer_regex(question, model_answer)

            if extracted_answer.lower() == target.lower():
                target_matches += 1
            elif extracted_answer in unknown_set:
                unknown_matches += 1
            else:
                other_matches += 1

            output_data = {
                "question": question,
                "target": target,
                "model_answer": model_answer,
                "extracted_answer": extracted_answer
            }
            json.dump(output_data, outfile, ensure_ascii=False)
            outfile.write("\n")

    print(f"Metrics for {input_file}:")
    print(f"  Accuracy (Target Matches): {target_matches / total_samples:.2f}")
    print(f"  Unknown Matches: {unknown_matches / total_samples:.2f}")
    print(f"  Other Matches: {other_matches / total_samples:.2f}")
    print(f"  Total Samples: {total_samples}")


def main():
    parser = argparse.ArgumentParser(description="Process and evaluate LLM outputs.")
    parser.add_argument("--language", required=True, choices=["en", "fr", "hi"], help="Language of the evaluation (en, fr, hi).")
    parser.add_argument("--input_file", help="Path to a single input JSON file.")
    parser.add_argument("--batch", action="store_true", help="Process multiple files in the batch mode.")
    parser.add_argument("--input_dir", default="data/", help="Directory containing input files for batch processing.")
    parser.add_argument("--output_dir", default="outputs/", help="Directory to save the converted and evaluated files.")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.batch:
        for i in range(1, 21):
            input_file = os.path.join(args.input_dir, f"evaluation_exp{i}_{args.language}.json")
            jsonl_file = os.path.join(args.output_dir, f"evaluation_exp{i}_{args.language}_converted.jsonl")
            evaluated_file = os.path.join(args.output_dir, f"evaluation_exp{i}_{args.language}_evaluated.jsonl")

            if not os.path.exists(input_file):
                print(f"File {input_file} not found. Skipping.")
                continue

            print(f"Processing file: {input_file}")
            convert_to_jsonl(input_file, jsonl_file)
            evaluate_file(jsonl_file, evaluated_file, args.language)
    else:
        if not args.input_file:
            print("Please provide --input_file for single-file processing.")
            return

        jsonl_file = os.path.join(args.output_dir, "converted.jsonl")
        evaluated_file = os.path.join(args.output_dir, "evaluated.jsonl")

        print(f"Processing single file: {args.input_file}")
        convert_to_jsonl(args.input_file, jsonl_file)
        evaluate_file(jsonl_file, evaluated_file, args.language)


if __name__ == "__main__":
    main()
