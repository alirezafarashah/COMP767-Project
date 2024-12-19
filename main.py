import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, default_data_collator, \
    DataCollatorForLanguageModeling
from datasets import load_dataset

import copy
import argparse
import logging
import os
from pathlib import Path

from data_utils import get_dataloaders

from train import train_loop


def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def save_model(output_dir):
    save_path = output_dir
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


if __name__ == '__main__':
    home_dir = Path.home()
    parser = argparse.ArgumentParser(description="Unlearn biases in large language models.")
    parser.add_argument("--model_name", type=str, default="unsloth/Meta-Llama-3.1-8B-Instruct",
                        help="The name of the pre-trained model.")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate for training.")
    parser.add_argument("--weights", type=float, nargs=3, default=[1.0, 0.25, 0.5],
                        help="Weights for kl_weight, unlearn_weight, and unk_weight (in that order).")
    parser.add_argument("--output_dir", type=str, default=home_dir / "scratch/models/",
                        help="The output directory.")
    parser.add_argument("--dataset", type=str, default="./mcq_stereotype_dataset.csv",
                        help="The output directory.")
    parser.add_argument("--log_dir", type=str, default="./log", help="Log dir to capture the output.")
    parser.add_argument("--language", type=str, choices=["en", "hi", "fr"], default="en",
                        help="Language for unlearning (options: en, hi, fr)")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, "training.log")
    logger = setup_logger(log_file)
    logger.info("Parsed Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")

    dataset = load_dataset("truthful_qa", 'generation', split="validation")
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    print("model loaded")
    seegull_ds = load_dataset("csv", data_files=args.dataset)
    seegull_ds = seegull_ds['train']
    pretrained_model = copy.deepcopy(model)
    pretrained_model.eval()
    print("pretrained model loaded")
    model.to("cuda:0")
    pretrained_model.to("cuda:1")
    print("model loaded on GPU")
    train_dataloader, train_unk_dataloader, normal_dataloader = get_dataloaders(tokenizer, dataset, seegull_ds,
                                                                                args.language)

    # Convert weights from string to float
    kl_weight, unlearn_weight, unk_weight = args.weights
    logger.info("Starting training loop")
    train_loop(
        model, pretrained_model, train_dataloader, train_unk_dataloader, normal_dataloader, logger,
        learning_rate=args.learning_rate, kl_weight=kl_weight, unlearn_weight=unlearn_weight, unk_weight=unk_weight
    )
    save_model(args.output_dir)
