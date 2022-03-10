"""Run a hyperparameter search on a RoBERTa model fine-tuned on BoolQ.

Example usage:
    python run_hyperparameter_search.py BoolQ/
"""
import argparse
import boolq
import data_utils
import finetuning_utils
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast
from transformers import TrainingArguments, Trainer




from ray.tune.suggest.bayesopt import BayesOptSearch
from ray import tune


parser = argparse.ArgumentParser(
    description="Run a hyperparameter search for finetuning a RoBERTa model on the BoolQ dataset."
)
parser.add_argument(
    "data_dir",
    type=str,
    help="Directory containing the BoolQ dataset. Can be downloaded from https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip.",
)

args = parser.parse_args()

# Since the labels for the test set have not been released, we will use half of the
# validation dataset as our test dataset for the purposes of this assignment.
train_df = pd.read_json(f"{args.data_dir}/train.jsonl", lines=True, orient="records")
val_df, test_df = train_test_split(
    pd.read_json(f"{args.data_dir}/val.jsonl", lines=True, orient="records"),
    test_size=0.5,
)

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
train_data = boolq.BoolQDataset(train_df, tokenizer)
val_data = boolq.BoolQDataset(val_df, tokenizer)
test_data = boolq.BoolQDataset(test_df, tokenizer)

## TODO: Initialize a transformers.TrainingArguments object here for use in
## training and tuning the model. Consult the assignment handout for some
## sample hyperparameter values.

training_args = TrainingArguments(
    output_dir="./models/",
    do_train=True,
    do_eval=True,
    per_gpu_train_batch_size=8,
    per_gpu_eval_batch_size=64,
    num_train_epochs=0.5, # due to time/computation constraints
    logging_steps=500,
    logging_first_step=True,
    save_steps=1000,
    evaluation_strategy = "epoch", # evaluate at the end of every epoch
    learning_rate=2e-5,
    weight_decay=0.01,
)

## TODO: Initialize a transformers.Trainer object and run a Bayesian
## hyperparameter search for at least 5 trials (but not too many) on the 
## learning rate. Hint: use the model_init() and
## compute_metrics() methods from finetuning_utils.py as arguments to
## Trainer(). Use the hp_space parameter in hyperparameter_search() to specify
## your hyperparameter search space. (Note that this parameter takes a function
## as its value.)
## Also print out the run ID, objective value,
## and hyperparameters of your best run.

model = finetuning_utils.model_init()
hyp_search = finetuning_utils.compute_metrics


trainer = Trainer(
    model,
    training_args,
    train_dataset=train_data,
    eval_dataset= val_data,
    tokenizer=tokenizer,
)

def temp_hyp_fn():
    temp_dict = {'learn': tune.uniform(1e-5,5e-5)}
    return temp_dict

trial = trainer.hyperparameter_search(hp_space = temp_hyp_fn,
                                      n_trials = 5,
                                      compute_objective = lambda : metrics['eval_accuracy'],
                                      search_alg= BayesOptSearch(metric = 'eval_loss', mode = 'min'))

