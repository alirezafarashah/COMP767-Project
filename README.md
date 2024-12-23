# Unlearning Geo-Cultural Stereotypes in Multilingual LLMs

## Abstract
As generative multilingual models become more widely used, most safety and fairness evaluations still focus on English-language resources, overlooking important cross-cultural factors. This limitation introduces issues of fairness and safety, particularly regarding Geo-Culturally Situated Stereotypes. In this work, we investigate debiasing methods aimed at unlearning these stereotypes by leveraging techniques such as gradient ascent (GA).

---

## Prerequisites

### Dependencies
- Python 3.8+

Install dependencies via:
```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Unlearning Biases
The `main.py` script is used to train a model to unlearn biases.

#### Arguments
| Argument          | Description                                  | Default                                   |
|-------------------|----------------------------------------------|-------------------------------------------|
| `--model_name`    | Name of the pre-trained model                | `unsloth/Meta-Llama-3.1-8B-Instruct`      |
| `--learning_rate` | Learning rate for unlearning                 | `1e-6`                                    |
| `--weights`       | Weights for `kl_weight`, `unlearn_weight`, and `unk_weight` | `[1.0, 0.25, 0.5]`                        |
| `--output_dir`    | Directory to save trained models             | `~/scratch/models/`                      |
| `--dataset`       | Path to the dataset                         | `./mcq_stereotype_dataset.csv`           |
| `--log_dir`       | Log directory                               | `./log`                                   |
| `--language`      | Language for unlearning (`en`, `hi`, `fr`)  | `en`                                      |

#### Example Command
```bash
python main.py --model_name unsloth/Meta-Llama-3.1-8B-Instruct --learning_rate 1e-6 --weights 1.0 0.25 0.5 --dataset ./mcq_stereotype_dataset.csv --language en
```

### 2. Collecting Model Responses
The `eval.py` script evaluates unlearned models on a dataset and collects responses.

#### Arguments
| Argument          | Description                                    | Required | Default                  |
|-------------------|------------------------------------------------|----------|--------------------------|
| `--model_dir`     | Directory of saved models                     | Yes      |                          |
| `--dataset`       | Path to the dataset                           | Yes      |                          |
| `--output_dir`    | Directory to save evaluation outputs          | No       | `./outputs/`            |
| `--filename`      | Output file name                              | No       | `unlearned_models_output_fr` |
| `--language`      | Language for evaluation (`en`, `hi`, `fr`)    | No       | `en`                    |

#### Example Command
```bash
python eval.py --model_dir ./saved_models/ --dataset ./mcq_stereotype_dataset.csv --output_dir ./outputs/ --filename unlearned_models_output_en --language en
```

### 3. Processing and Evaluating Responses
The `evaluation.py` script processes the collected responses and evaluates them.

#### Arguments
| Argument         | Description                                    | Required | Default                  |
|------------------|------------------------------------------------|----------|--------------------------|
| `--language`     | Language for evaluation (`en`, `fr`, `hi`)     | Yes      |                          |
| `--input_file`   | Path to a single input JSON file               | No       |                          |
| `--batch`        | Process multiple files in batch mode           | No       |                          |
| `--input_dir`    | Directory containing input files for batch     | No       | `data/`                 |
| `--output_dir`   | Directory to save the evaluation results       | No       | `outputs/`              |

#### Example Command
```bash
python evaluation.py --language en --input_file ./outputs/unlearned_models_output_en.json
```

