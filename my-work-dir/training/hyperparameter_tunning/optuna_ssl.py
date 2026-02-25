import csv
import os
import optuna

from torch.utils.data import DataLoader

from config.config import load_config
from experiments.run_ssl import run_ssl_experiment

def run_ssl_optuna(cfg):

    # Objective function to be optimized
    def objective(trial):

        cfg = load_config("config/ssl_hyptun.yaml")

        # --------------------------------------
        # Suggest values for hyperparameters
        # --------------------------------------

        # Model dimensionalities
        cfg["model"]["backbone_dim"] = trial.suggest_categorical("backbone_dim", [512, 2048])
        cfg["model"]["hidden_dim"] = trial.suggest_int("hidden_dim", 256, 2048, step=256)
        cfg["model"]["projection_dim"] = trial.suggest_int("projection_dim", 512, 2048, step=256)

        # Model parameters
        cfg["training"]["learning_rate"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        cfg["training"]["temperature"] = trial.suggest_float("temperature", 0.05, 0.5)
        cfg["training"]["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

        # Set trial info
        cfg["experiment_name"] = f"ssl_optuna_trial"
        cfg["experiment_version"] = f"1.{trial.number}"

        # ---------------------------------------
        # Build model
        # ---------------------------------------
        model, history, stop_epoch = run_ssl_experiment(cfg, trial)

        # ---------------------------------------
        # Return metric to maximize
        # ---------------------------------------
        final_acc = history["accuracy"].iloc[-1]
        final_epoch = stop_epoch
        final_uniformity = history["uniformity"].iloc[-1]
        final_alignment = history["alignment"].iloc[-1]
        final_loss = history["contrastive_loss"].iloc[-1]
        total_seconds = history["time"].sum() 
        hours = total_seconds // 3600 
        minutes = (total_seconds % 3600) // 60 
        seconds = total_seconds % 60

        trial.set_user_attr("final_accuracy", final_acc)
        trial.set_user_attr("epochs", final_epoch)
        trial.set_user_attr("final_uniformity", final_uniformity)
        trial.set_user_attr("final_alignment", final_alignment)
        trial.set_user_attr("final_loss", final_loss)
        trial.set_user_attr("training_time", f"{hours}:{minutes}:{seconds}")
        trial.set_user_attr("history_path", f"{cfg['paths']['output_dir']}/ssl_history_{cfg['experiment_name']}_{cfg['experiment_version']}.csv")

        return final_acc

    study = optuna.create_study(
        study_name=cfg["experiment_name"],
        direction="maximize",
        storage=f"sqlite:///{cfg['paths']['output_dir']}/optuna_ssl{cfg['experiment_version']}.db", 
        load_if_exists=True, 
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20), 
    )
    study.optimize(objective, n_trials=cfg["optuna"]["n_trials"])

    print("Best trial:", study.best_trial.params)