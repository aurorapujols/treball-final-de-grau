import argparse
import time

# Experiment entry points
from experiments.run_ssl import run_ssl_experiment
from experiments.run_plots import plot_model_results
from experiments.run_classifier import train_classifiers
from config.config import load_config
from training.hyperparameter_tunning.optuna_ssl import run_ssl_optuna
from data.datasets import get_dataset_split

def main():
    
    parser = argparse.ArgumentParser(description="Meteor Representation Learning Pipeline")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["ssl", "ssl_augs", "ssl_architecture", "ssl_hyptun", "ssl_final_model", "plot_results", "train_classifiers", "temp"],
        help="Which experiment to run"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    # Load YAML config
    cfg = load_config(args.config)

    # Dispatch to the correct experiment
    if args.task == "ssl":
        run_ssl_experiment(cfg)

    elif args.task == "ssl_augs":
        augs = [[0], [1], [2], [3], [0,1], [1,2], [2,3], [0,2], [1,3], [0,3], [0,1,2], [0,1,3], [0,2,3], [1,2,3], [0,1,2,3]]  # execute all augmentations in one run
        for version, augs_idx in enumerate(augs):
            run_ssl_experiment(cfg, add_version=version+1, augs_idx=augs_idx)
    
    elif args.task == "ssl_architecture":
        run_ssl_experiment(cfg)

    elif args.task == "ssl_hyptun":
        run_ssl_optuna(cfg)

    elif args.task == "plot_results":
        plot_model_results(cfg)

    elif args.task == "ssl_final_model":
        run_ssl_experiment(cfg)
    
    elif args.task == "train_classifiers":
        train_classifiers(cfg)

    elif args.task == "temp":
        get_dataset_split(full_dataset_csv_path=cfg['full_dataset_path'], output_path=cfg['output_path'])


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time() - start_time

    # Calculate hours, minutes, and seconds
    hours, rem = divmod(end_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"Elapsed time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
