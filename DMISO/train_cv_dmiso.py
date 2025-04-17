import argparse
import json
import logging
import os
import time
from datetime import datetime
import gc  # Added for memory clearing

import numpy as np
import optuna
from optuna.samplers import TPESampler
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import CometLogger, CSVLogger
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import random_split, DataLoader, Subset
from torch_geometric.data import DataLoader as GraphDataLoader
import psutil  # For CPU memory tracking

from source.pytorch.datasets import (
    MiTarDataset, CustomDataset, PredictionDataset, HelwakDataset,
    MiTarGraphDataset, CustomGraphDataset
)
from source.pytorch.modules import MiTarModule

CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/")
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

def get_max_memory_usage():
    """Returns the maximum memory usage in MB. Uses CUDA if available, else CPU."""
    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        torch.cuda.reset_max_memory_allocated()
        return max_memory
    else:
        process = psutil.Process(os.getpid())
        max_memory = process.memory_info().rss / (1024 ** 2)
        return max_memory

def main(args):
    overall_start = time.time()
    L.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set up logging directory
    log_root = args.log_dir  # e.g., 'dmiso_logs'
    objective_log_dir = os.path.join(log_root, "objective")  # e.g., 'dmiso_logs/objective'
    os.makedirs(log_root, exist_ok=True)
    os.makedirs(objective_log_dir, exist_ok=True)

    # Load dataset
    if args.dataset == "MiTar":
        if args.model == "GraphTar":
            dataset = MiTarGraphDataset(args.input_data_path, args.word2vec_model_dir)
        else:
            dataset = MiTarDataset(
                args.input_data_path,
                n_encoding=tuple([float(n.strip()) for n in args.n_encoding.split(",")])
            )
    elif args.dataset == "Custom":
        if args.model == "GraphTar":
            dataset = CustomGraphDataset(args.input_data_path, args.word2vec_model_dir)
        else:
            dataset = CustomDataset(
                args.input_data_path,
                canonical_only=args.canonical_only,
                strip_n=args.strip_n,
                n_encoding=tuple([float(n.strip()) for n in args.n_encoding.split(",")])
            )
    elif args.dataset == "Helwak":
        dataset = HelwakDataset(
            args.input_data_path,
            strip_n=args.strip_n,
            n_encoding=tuple([float(n.strip()) for n in args.n_encoding.split(",")]),
            icshape=args.icshape
        )
    else:
        raise ValueError(f"Invalid dataset name {args.dataset}")

    dataloader = DataLoader
    if args.model == "GraphTar":
        dataloader = GraphDataLoader

    # Construct model configuration
    if args.model == "CNNSequenceModel":
        model_config = {
            'incorporate_type': args.incorporate_type,
            'num_filters': 12, 'kernel_size': 8, 'dropout_rate': 0.1,
            'max_mirna_len': dataset.max_mirna_len,
            'max_target_len': dataset.max_target_len,
            'n_encoding': dataset.n_encoding,
            'mirna_dim': dataset.mirna_dim,
            'target_dim': dataset.target_dim
        }
    elif args.model == "HybridSequenceModel":
        model_config = {
            'num_filters': 12, 'kernel_size': 8,
            'dropout_rate': 0.1, 'rnn_units': 32
        }
    elif args.model == "CNNTransformerModel":
        model_config = {
            'num_filters': 12, 'kernel_size': 8,
            'dropout_rate': 0.1, 'num_heads': 3, 'hidden_size': 6
        }
    elif args.model == "DMISO":
        model_config = {
            'max_mirna_len': dataset.max_mirna_len,
            'max_target_len': dataset.max_target_len,
            'n_encoding': dataset.n_encoding,
            'l1_lambda': 0.01,
            'mirna_dim': dataset.mirna_dim,
            'target_dim': dataset.target_dim
        }
    elif args.model == "MiTar":
        model_config = {
            'n_embeddings': len(dataset.n_encoding),
            'max_mirna_len': dataset.max_mirna_len,
            'max_target_len': dataset.max_target_len
        }
    elif args.model == "TransPHLA":
        model_config = {
            'n_layers': 1, 'd_model': 64, 'n_heads': 9,
            'd_ff': 512, 'd_k': 64, 'd_v': 64,
            'dropout_rate': 0.1,
            'max_mirna_len': dataset.max_mirna_len,
            'max_target_len': dataset.max_target_len,
        }
    elif args.model == "TEC-miTarget":
        model_config = {
            'max_mirna_len': dataset.max_mirna_len,
            'max_target_len': dataset.max_target_len,
            'input_dim': 512, 'projection_dim': 256, 'n_heads': 1,
            'n_layers': 6, 'dropout': 0, 'kernal_size': 9,
            'p0': 0.5, 'gamma_init': 0
        }
    elif args.model == "GraphTar":
        model_config = None
    else:
        raise ValueError(f"Invalid model name {args.model}")

    try:
        labels = [sample[2] for sample in dataset]
    except Exception as e:
        log.error("Error extracting labels: %s", e)
        labels = [int(data.y) for data in dataset]

    skf = StratifiedKFold(n_splits=args.num_cv_folds, shuffle=True, random_state=args.seed)
    all_fold_results = {}
    exported_splits = {}

    process = psutil.Process()  # Initialize process monitoring for the entire script

    for fold, (train_val_indices, test_indices) in enumerate(skf.split(dataset, labels)):
        fold_start = time.time()
        log.info(f"Starting fold {fold}")

        train_val_indices = np.array(train_val_indices)
        test_indices = np.array(test_indices)

        rng = np.random.RandomState(args.seed + fold)
        total_train_val = len(train_val_indices)
        val_size = int(args.val_ratio * total_train_val)
        shuffled = train_val_indices.copy()
        rng.shuffle(shuffled)
        val_indices = shuffled[:val_size]
        train_indices = shuffled[val_size:]
        
        exported_splits[f"fold_{fold}"] = {
            "train_indices": train_indices.tolist(),
            "val_indices": val_indices.tolist(),
            "test_indices": test_indices.tolist()
        }

        train_val_set = Subset(dataset, train_val_indices.tolist())
        trial_results = []

        def objective(trial):
            total = len(train_val_set)
            val_size_trial = int(0.1 * total)
            train_size_trial = total - val_size_trial
            train_subset_trial, val_subset_trial = random_split(train_val_set, [train_size_trial, val_size_trial])

            opt_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'Adadelta'])
            dropout_rate = trial.suggest_categorical('dropout_rate', [0.1, 0.2, 0.3, 0.4, 0.5])
            trial_batch_size = trial.suggest_categorical('batch_size', [10, 30, 50, 100, 200])
            
            opt_config = {}
            if opt_name == 'Adam':
                lr = trial.suggest_categorical('lr_adam', [0.2, 0.1, 0.05, 0.01, 0.005, 0.001])
                opt_config['lr'] = lr
            elif opt_name == 'SGD':
                lr = trial.suggest_categorical('lr_sgd', [0.2, 0.1, 0.05, 0.01, 0.005, 0.001])
                momentum = trial.suggest_float('momentum', 0.0, 0.9)
                opt_config['lr'] = lr
                opt_config['momentum'] = momentum
            elif opt_name == 'Adadelta':
                lr = trial.suggest_float('lr_adadelta', 0.001, 1.0)
                rho = trial.suggest_float('rho', 0.9, 0.99)
                opt_config['lr'] = lr
                opt_config['rho'] = rho
            
            opt_config['weight_decay'] = args.weight_decay

            # Log hyperparameters
            log.info(f"Fold {fold}, Trial {trial.number} - Hyperparameters: optimizer={opt_name}, dropout_rate={dropout_rate}, batch_size={trial_batch_size}, opt_config={opt_config}")

            if torch.cuda.is_available():
                torch.cuda.reset_max_memory_allocated()

            train_loader = dataloader(train_subset_trial, batch_size=trial_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            val_loader = dataloader(val_subset_trial, batch_size=trial_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

            current_model_config = model_config.copy() if model_config is not None else {}
            if 'dropout_rate' in current_model_config:
                current_model_config['dropout_rate'] = dropout_rate

            model = MiTarModule(
                model_name=args.model,
                model_config=current_model_config,
                opt_name=opt_name,
                opt_config=opt_config
            )

            checkpoint_callback = ModelCheckpoint(
                filename=f"fold{fold}_trial{trial.number}-{{epoch:02d}}-{{val_acc:.4f}}",
                save_top_k=1,
                save_weights_only=True,
                monitor="val_acc",
                mode="max"
            )
            simple_logger = CSVLogger(save_dir=objective_log_dir, name=f"fold{fold}_trial{trial.number}")
            trainer = L.Trainer(
                logger=simple_logger,
                default_root_dir=os.path.join(CHECKPOINT_PATH, f"{args.model}/fold{fold}/trial_{trial.number}"),
                accelerator="auto",
                devices=1,
                inference_mode=False,
                max_epochs=args.epochs,
                callbacks=[
                    checkpoint_callback,
                    LearningRateMonitor(logging_interval="epoch"),
                    EarlyStopping(monitor='val_acc', min_delta=0.001, patience=args.patience, verbose=False, mode='max')
                ]
            )

            # Detailed monitoring for trial
            start_time = time.time()
            mem_before = process.memory_info().rss / (1024 ** 2)
            log.info(f"Fold {fold}, Trial {trial.number} - Memory before training: {mem_before:.2f} MB")
            
            trainer.fit(model, train_loader, val_loader)
            
            end_time = time.time()
            time_taken = end_time - start_time
            mem_after = process.memory_info().rss / (1024 ** 2)
            cpu_percent = psutil.cpu_percent(interval=None)
            max_memory_mb = get_max_memory_usage()
            
            log.info(f"Fold {fold}, Trial {trial.number} - Training time: {time_taken:.2f} sec")
            log.info(f"Fold {fold}, Trial {trial.number} - Memory after training: {mem_after:.2f} MB")
            log.info(f"Fold {fold}, Trial {trial.number} - Peak CPU usage: {cpu_percent:.2f}%")
            log.info(f"Fold {fold}, Trial {trial.number} - Max memory usage: {max_memory_mb:.2f} MB")

            trial_value = checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score is not None else float('-inf')
            val_metrics = trainer.validate(model, val_loader)
            if isinstance(val_metrics, list) and len(val_metrics) > 0:
                val_metrics = val_metrics[0]
            trial_info = {
                "trial_number": trial.number,
                "hyperparameters": {
                    "optimizer": opt_name,
                    "dropout_rate": dropout_rate,
                    "batch_size": trial_batch_size,
                    **opt_config
                },
                "validation_acc": trial_value,
                "validation_metrics": val_metrics,
                "max_memory_usage_mb": max_memory_mb,
                "timestamp": datetime.now().isoformat(),
                "best_model_path": checkpoint_callback.best_model_path,  # Added for pretrained model
                "training_time_sec": time_taken,
                "cpu_percent": cpu_percent,
                "memory_before_mb": mem_before,
                "memory_after_mb": mem_after
            }
            trial_results.append(trial_info)

            # Clear memory after trial
            del model, trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return trial_value

        # Modified to use SQLite database for Optuna study
        storage_url = f"sqlite:///dmiso_study_fold_{fold}.db"
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=args.seed + fold), storage=storage_url)
        study.optimize(objective, n_trials=args.num_trials)
        best_params = study.best_trial.params
        log.info(f"Best params for fold {fold}: {best_params}")

        best_trial_metrics = None
        for t in trial_results:
            if t["trial_number"] == study.best_trial.number:
                best_trial_metrics = t.get("validation_metrics", {})
                break

        fold_summary = {
            "trial_results": trial_results,
            "best_trial": {
                "trial_number": study.best_trial.number,
                "hyperparameters": best_params,
                "best_value": study.best_trial.value,
                "validation_metrics": best_trial_metrics
            }
        }

        current_splits = exported_splits[f"fold_{fold}"]
        train_subset = Subset(dataset, current_splits["train_indices"])
        val_subset = Subset(dataset, current_splits["val_indices"])
        test_subset = Subset(dataset, current_splits["test_indices"])

        final_batch_size = best_params["batch_size"]
        train_loader = dataloader(train_subset, batch_size=final_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        val_loader = dataloader(val_subset, batch_size=final_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        # Adjusted test loader for CPU with a larger batch size
        test_batch_size = 64  # Adjust based on your system's memory capacity
        test_loader = dataloader(test_subset, batch_size=test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

        final_model_config = model_config.copy() if model_config is not None else {}
        if 'dropout_rate' in final_model_config:
            final_model_config['dropout_rate'] = best_params['dropout_rate']

        final_opt_name = best_params['optimizer']
        final_opt_config = {k: v for k, v in best_params.items() if k.startswith('lr_') or k in ['momentum', 'rho']}
        if 'lr_' + final_opt_name.lower() in final_opt_config:
            final_opt_config['lr'] = final_opt_config.pop('lr_' + final_opt_name.lower())
        final_opt_config['weight_decay'] = args.weight_decay

        # Log final hyperparameters
        log.info(f"Fold {fold} - Final model hyperparameters: optimizer={final_opt_name}, dropout_rate={best_params['dropout_rate']}, batch_size={final_batch_size}, opt_config={final_opt_config}")

        # Get the best model's checkpoint path
        best_trial_number = study.best_trial.number
        best_model_path = None
        for t in trial_results:
            if t["trial_number"] == best_trial_number:
                best_model_path = t["best_model_path"]
                break
        if best_model_path is None:
            raise ValueError("Best model path not found for the best trial")

        final_model = MiTarModule(
            model_name=args.model,
            model_config=final_model_config,
            opt_name=final_opt_name,
            opt_config=final_opt_config
        )

        # Load the best trial's weights into the final model
        checkpoint = torch.load(best_model_path)
        final_model.load_state_dict(checkpoint["state_dict"])

        save_name = f"{args.model}/fold-{fold}"
        final_logger = (CometLogger(
            api_key="-",
            project_name="DMISO_on_CLIP_data",  # Updated project name for DMISO
            workspace="-"
        ) if args.comet_logging else CSVLogger(save_dir=log_root, name=save_name))
        final_checkpoint_callback = ModelCheckpoint(
            filename="{epoch:02d}-{val_acc:.4f}",
            save_top_k=1,
            save_weights_only=True,
            monitor="val_acc",
            mode="max"
        )
        final_trainer = L.Trainer(
            logger=final_logger,
            default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),
            accelerator="auto",
            devices="auto",
            inference_mode=False,
            max_epochs=args.epochs,
            callbacks=[
                final_checkpoint_callback,
                LearningRateMonitor(logging_interval="epoch"),
                EarlyStopping(monitor='val_acc', min_delta=0.001, patience=args.patience, verbose=False, mode='max')
            ]
        )

        # Detailed monitoring for final training
        log.info(f"Training final model for fold {fold} with best hyperparameters...")
        start_time = time.time()
        mem_before = process.memory_info().rss / (1024 ** 2)
        log.info(f"Fold {fold} - Memory before final training: {mem_before:.2f} MB")
        
        if torch.cuda.is_available():
            torch.cuda.reset_max_memory_allocated()
        final_trainer.fit(final_model, train_loader, val_loader)
        
        end_time = time.time()
        time_taken = end_time - start_time
        mem_after = process.memory_info().rss / (1024 ** 2)
        cpu_percent = psutil.cpu_percent(interval=None)
        max_memory_mb = get_max_memory_usage()
        
        log.info(f"Fold {fold} - Final training time: {time_taken:.2f} sec")
        log.info(f"Fold {fold} - Memory after final training: {mem_after:.2f} MB")
        log.info(f"Fold {fold} - Peak CPU usage during final training: {cpu_percent:.2f}%")
        log.info(f"Fold {fold} - Max memory usage during final training: {max_memory_mb:.2f} MB")

        # Switch to CPU for testing
        log.info(f"Switching to CPU for testing fold {fold}...")
        final_model = final_model.cpu()  # Move model to CPU
        
        # Create a new Trainer instance for CPU testing
        test_trainer = L.Trainer(
            accelerator='cpu',  # Explicitly use CPU
            devices=1,
            inference_mode=False  # No gradients needed for testing
        )
        
        # Detailed monitoring for testing
        log.info(f"Testing final model for fold {fold} on CPU...")
        start_time = time.time()
        mem_before_test = process.memory_info().rss / (1024 ** 2)
        log.info(f"Fold {fold} - Memory before testing: {mem_before_test:.2f} MB")
        
        # Free up memory before testing
        del train_loader, val_loader
        test_results = test_trainer.test(final_model, test_loader)
        
        end_time = time.time()
        time_taken_test = end_time - start_time
        mem_after_test = process.memory_info().rss / (1024 ** 2)
        cpu_percent_test = psutil.cpu_percent(interval=None)
        max_memory_mb_test = get_max_memory_usage()
        
        log.info(f"Fold {fold} - Testing time: {time_taken_test:.2f} sec")
        log.info(f"Fold {fold} - Memory after testing: {mem_after_test:.2f} MB")
        log.info(f"Fold {fold} - Peak CPU usage during testing: {cpu_percent_test:.2f}%")
        log.info(f"Fold {fold} - Max memory usage during testing: {max_memory_mb_test:.2f} MB")
        
        if isinstance(test_results, list) and len(test_results) > 0:
            final_metrics = test_results[0]
        else:
            final_metrics = test_results
        fold_summary["final_metrics"] = final_metrics
        fold_summary["final_training_time_sec"] = time_taken
        fold_summary["final_testing_time_sec"] = time_taken_test
        fold_summary["final_cpu_percent_training"] = cpu_percent
        fold_summary["final_cpu_percent_testing"] = cpu_percent_test
        fold_summary["final_memory_before_mb"] = mem_before
        fold_summary["final_memory_after_mb"] = mem_after
        fold_summary["final_memory_before_test_mb"] = mem_before_test
        fold_summary["final_memory_after_test_mb"] = mem_after_test
        fold_summary["final_max_memory_usage_mb"] = max_memory_mb
        fold_summary["final_max_memory_usage_test_mb"] = max_memory_mb_test
        fold_summary["final_hyperparameters"] = {
            "optimizer": final_opt_name,
            "dropout_rate": best_params['dropout_rate'],
            "batch_size": final_batch_size,
            **final_opt_config
        }

        fold_end = time.time()
        fold_time = fold_end - fold_start
        fold_summary["fold_execution_time_sec"] = fold_time
        log.info(f"Fold {fold} executed in {fold_time:.2f} seconds.")

        fold_results_file = os.path.join(objective_log_dir, f"fold{fold}_results.json")
        with open(fold_results_file, "w") as f:
            json.dump(fold_summary, f, indent=4)
        all_fold_results[f"fold_{fold}"] = fold_summary

        # Clear memory after fold
        del final_model, final_trainer, test_trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(os.path.join(log_root, "all_folds_summary.json"), "w") as f:
        json.dump(all_fold_results, f, indent=4)
    with open(os.path.join(log_root, "exported_splits.json"), "w") as f:
        json.dump(exported_splits, f, indent=4)
    overall_end = time.time()
    overall_time = overall_end - overall_start
    log.info(f"Total execution time: {overall_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=500, help='Maximum number of epochs to train for')  # Changed to 500
    parser.add_argument('--batch-size', type=int, default=32, help='Default batch size (overridden by hyperparameter search)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default, may be tuned via Optuna)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay, use 0 for no weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Number of epochs to wait before early stopping')
    parser.add_argument('--num-cv-folds', type=int, default=5, help='Number of folds used in cross-validation')
    parser.add_argument('--num-trials', type=int, default=100, help='Number of Optuna trials per fold')  # Changed to 100
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Ratio for validation split from train_val set')
    parser.add_argument('--model', type=str, default='DMISO',  # Updated default to DMISO
                        choices=['CNNSequenceModel', 'DMISO', 'MiTar', 'TransPHLA', 'TEC-miTarget', 'GraphTar'],
                        help='Name of the model to load')
    parser.add_argument('--comet-logging', action='store_true', help='Whether to log to Comet.ml')
    parser.add_argument('--dataset', type=str, default='Custom', help='Dataset to use',
                        choices=['MiTar', 'Custom', 'Helwak'])
    parser.add_argument('--input-data-path', type=str,
                        default='/projects/mirbench/work/DLmiRTPred-withoptuna/data/mitar/allCLIP_final.txt',
                        help='Path to input data')
    parser.add_argument('--prediction-data-path', type=str, default=None, help='Path to prediction data')
    parser.add_argument('--canonical-only', action='store_true', help='Whether to use only canonical miRNA-target pairs')
    parser.add_argument('--incorporate-type', action='store_true', help='Whether to incorporate miRNA-target interaction type')
    parser.add_argument('--strip-n', action='store_true', help='Whether to strip padding Ns from sequences')
    parser.add_argument('--n-encoding', type=str, default='1,0,0,0,0', help='Encoding for N in sequences')
    parser.add_argument('--icshape', action='store_true', help='Whether to use icSHAPE structure data')
    parser.add_argument('--word2vec-model-dir', type=str, default='data/mitar/word2vec_models', help='Directory containing word2vec models')
    parser.add_argument('--num-workers', type=int, default=40, help='Number of workers for data loading')
    parser.add_argument('--log-dir', type=str, default='dmiso_log_CLIP', help='Directory for log files and checkpoints')

    args = parser.parse_args()
    print("Arguments: ", args)

    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()
    main(args)
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    stats.dump_stats("profile_results.prof")
