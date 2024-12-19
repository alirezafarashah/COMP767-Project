import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, default_data_collator, \
    DataCollatorForLanguageModeling


def get_dataloaders(tokenizer, dataset, seegull_ds, language='en'):
    EOS = tokenizer.eos_token
    # Tokenize the dataset with padding/truncation
    tokenized_dataset = seegull_ds.map(preprocess_mcq, batched=True, batch_size=100,
                                       remove_columns=seegull_ds.column_names,
                                       fn_kwargs={"tokenizer": tokenizer, "EOS": EOS, "language": language})
    tokenized_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "start_locs"]
    )

    unk_dataset = seegull_ds.map(preprocess_mcq_unk, batched=True, batch_size=100,
                                 remove_columns=seegull_ds.column_names,
                                 fn_kwargs={"tokenizer": tokenizer, "EOS": EOS, "language": language})
    unk_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "start_locs"]
    )
    normalized_dataset = dataset.map(preproccess_truthfulqa, batched=True, batch_size=100,
                                     remove_columns=dataset.column_names,
                                     fn_kwargs={"tokenizer": tokenizer, "EOS": EOS})
    normalized_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "start_locs"]
    )

    # Set up DataLoader with default_data_collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = torch.utils.data.DataLoader(
        tokenized_dataset, batch_size=4, collate_fn=data_collator
    )
    normal_dataloader = torch.utils.data.DataLoader(
        normalized_dataset, batch_size=4, collate_fn=data_collator
    )
    train_unk_dataloader = torch.utils.data.DataLoader(
        unk_dataset, batch_size=4, collate_fn=data_collator
    )
    return train_dataloader, train_unk_dataloader, normal_dataloader


def preproccess_truthfulqa(examples, tokenizer, EOS):
    """
    Input: Dict[List]
    Output: Dict[List]
    """
    results = {"input_ids": [], "attention_mask": [], "start_locs": []}
    for q, a in zip(examples['question'], examples['best_answer']):
        text = f"Question: {q}\n Answer: {a}" + EOS
        tokenized = tokenizer(text, truncation=True, padding="longest")
        results["input_ids"].append(tokenized["input_ids"])
        results["attention_mask"].append(tokenized["attention_mask"])
        # Calculate start idx for answer
        test_text = f"### Question: {q}\n ### Answer: "
        test_tokenized = tokenizer(
            test_text, truncation=False, padding=False)
        results["start_locs"].append(len(test_tokenized["input_ids"]) - 1)

    return results


def preprocess_mcq(examples, tokenizer, EOS, language='en'):
    """
    Input: Dict[List]
    Output: Dict[List]
    """
    results = {"input_ids": [], "attention_mask": [], "start_locs": []}

    for row in zip(*examples.values()):
        row_data = dict(zip(examples.keys(), row))
        q = row_data['MCQ_Question']
        correct_option_key = row_data['Correct_Option']
        a = row_data[correct_option_key]
        if language == 'en':
            text = f"Question: {q}\n Answer: {a}" + EOS
        elif language == 'fr':
            text = f"Question: {q}\n Réponse: {a}" + EOS
        # print(text)
        else:
            text = f"सवाल: {q}\n जवाब: {a}" + EOS
        # print(text)
        tokenized = tokenizer(text, truncation=True, padding="longest")
        results["input_ids"].append(tokenized["input_ids"])
        results["attention_mask"].append(tokenized["attention_mask"])
        # Calculate start idx for answer
        if language == 'en':
            test_text = f"Question: {q}\n Answer: "
        elif language == 'fr':
            test_text = f"Question: {q}\n Réponse: "
        # print(text)
        else:
            test_text = f"सवाल: {q}\n जवाब: "
        # test_text = f"Question: {q}\n Answer: "
        test_tokenized = tokenizer(
            test_text, truncation=False, padding=False)
        results["start_locs"].append(len(test_tokenized["input_ids"]) - 1)

    return results


def preprocess_mcq_unk(examples, tokenizer, EOS, language='en'):
    import random
    """
    Input: Dict[List]
    Output: Dict[List]
    """
    results = {"input_ids": [], "attention_mask": [], "start_locs": []}
    unknown_options = {"en": [
        "Cannot be determined",
        "Unknown",
        "Not enough information",
        "Unclear", ],

        "fr": [
            "Impossible à déterminer",
            "Inconnu",
            "Pas assez d'informations",
            "Peu clair", ],

        "hi": [
            "निर्धारित नहीं किया जा सकता",
            "अज्ञात",
            "पर्याप्त जानकारी नहीं है",
            "अस्पष्ट", ],
    }
    for row in zip(*examples.values()):
        row_data = dict(zip(examples.keys(), row))
        q = row_data['MCQ_Question']
        unknown_option = random.choice(unknown_options[language])
        if language == 'en':
            text = f"Question: {q}\n Answer: {unknown_option}" + EOS
        elif language == 'fr':
            text = f"Question: {q}\n Réponse: {unknown_option}" + EOS
        # print(text)
        else:
            text = f"सवाल: {q}\n जवाब: {unknown_option}" + EOS
        tokenized = tokenizer(text, truncation=True, padding="longest")
        results["input_ids"].append(tokenized["input_ids"])
        results["attention_mask"].append(tokenized["attention_mask"])
        # Calculate start idx for answer
        results["start_locs"].append(0)

    return results
