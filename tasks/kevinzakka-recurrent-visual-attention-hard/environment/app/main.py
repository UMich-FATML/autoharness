import argparse

import numpy as np

import modules
from config import NUM_CLASSES, RIDGE_REGULARIZATION
from data_loader import build_medium_split, candidate_locations, medium_dev_specs, medium_train_specs
from model import attended_state, initial_state
from utils import (
    accuracy,
    fit_linear_regression,
    fit_ridge_classifier,
    load_hidden_specs,
    predict_labels,
    snap_to_candidates,
    write_results,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "hidden_test"], default="dev")
    parser.add_argument("--hidden-module", default=None)
    parser.add_argument("--output", default="/app/results.json")
    args = parser.parse_args()

    train_split = build_medium_split(medium_train_specs())
    eval_specs = medium_dev_specs() if args.split == "dev" else load_hidden_specs(args.hidden_module)
    eval_split = build_medium_split(eval_specs)
    candidates = candidate_locations()

    first_states = np.stack([initial_state(sample["image"]) for sample in train_split])
    target_locations = np.stack([sample["location"] for sample in train_split])
    location_weights, location_bias = fit_linear_regression(first_states, target_locations, RIDGE_REGULARIZATION)

    class_features = np.stack(
        [attended_state(sample["image"], state, sample["location"]) for sample, state in zip(train_split, first_states)]
    )
    train_labels = np.asarray([sample["digit"] for sample in train_split], dtype=int)
    classifier = fit_ridge_classifier(class_features, train_labels, NUM_CLASSES, RIDGE_REGULARIZATION)

    predictions = []
    predicted_positions = []
    location_distances = []
    for sample in eval_split:
        first_state = initial_state(sample["image"])
        raw_location = np.asarray(modules.location_network(first_state, location_weights, location_bias), dtype=float)
        snapped_location, snapped_index = snap_to_candidates(raw_location, candidates)
        second_state = attended_state(sample["image"], first_state, snapped_location)
        prediction = int(predict_labels(np.asarray([second_state]), classifier)[0])

        predictions.append(prediction)
        predicted_positions.append(snapped_index)
        location_distances.append(float(np.linalg.norm(snapped_location - sample["location"])))

    eval_labels = np.asarray([sample["digit"] for sample in eval_split], dtype=int)
    eval_accuracy = accuracy(eval_labels, np.asarray(predictions, dtype=int))
    position_accuracy = accuracy(
        np.asarray([sample["position_index"] for sample in eval_split], dtype=int),
        np.asarray(predicted_positions, dtype=int),
    )

    payload = {
        "accuracy": eval_accuracy,
        "error_rate": 1.0 - eval_accuracy,
        "eval_examples": int(len(eval_split)),
        "mean_location_distance": float(np.mean(location_distances)),
        "position_accuracy": position_accuracy,
        "split": args.split,
        "train_examples": int(len(train_split)),
    }
    write_results(args.output, payload)


if __name__ == "__main__":
    main()
