import argparse

# Experiment entry points
from experiments.run_ssl import run_ssl_experiment
from experiments.run_plots import run_plot3d_embeddings
from config.config import load_config
from training.hyperparameter_tunning.optuna_ssl import run_ssl_optuna

def main():
    parser = argparse.ArgumentParser(description="Meteor Representation Learning Pipeline")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["ssl", "ssl_augs", "ssl_architecture", "ssl_hyptun", "plot_3d_embeddings_vgg"],
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

    elif args.task == "plot_3d_embeddings_vgg":
        run_plot3d_embeddings(cfg)

    # elif args.task == "classifier":
    #     run_classifier_experiment(cfg)

    # elif args.task == "clustering":
    #     run_clustering_experiment(cfg)


if __name__ == "__main__":
    main()
