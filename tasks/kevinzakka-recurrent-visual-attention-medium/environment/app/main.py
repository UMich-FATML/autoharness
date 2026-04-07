import argparse

import numpy as np

from config import GLIMPSE_DIM, NUM_CLASSES, RIDGE_REGULARIZATION, STATE_DIM
from data_loader import build_medium_split, medium_dev_specs, medium_train_specs
from model import medium_state
from utils import accuracy, fit_ridge_classifier, load_hidden_specs, predict_labels, write_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "hidden_test"], default="dev")
    parser.add_argument("--hidden-module", default=None)
    parser.add_argument("--output", default="/app/results.json")
    args = parser.parse_args()

    train_split = build_medium_split(medium_train_specs())
    eval_specs = medium_dev_specs() if args.split == "dev" else load_hidden_specs(args.hidden_module)
    eval_split = build_medium_split(eval_specs)

    train_features = np.stack([medium_state(sample["image"], sample["location"]) for sample in train_split])
    train_labels = np.asarray([sample["digit"] for sample in train_split], dtype=int)
    eval_features = np.stack([medium_state(sample["image"], sample["location"]) for sample in eval_split])
    eval_labels = np.asarray([sample["digit"] for sample in eval_split], dtype=int)

    weights = fit_ridge_classifier(train_features, train_labels, NUM_CLASSES, RIDGE_REGULARIZATION)
    predictions = predict_labels(eval_features, weights)
    eval_accuracy = accuracy(eval_labels, predictions)

    payload = {
        "accuracy": eval_accuracy,
        "error_rate": 1.0 - eval_accuracy,
        "eval_examples": int(len(eval_split)),
        "glimpse_dim": GLIMPSE_DIM,
        "split": args.split,
        "state_dim": STATE_DIM,
        "train_examples": int(len(train_split)),
    }
    write_results(args.output, payload)


if __name__ == "__main__":
    main()
