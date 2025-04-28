import argparse
import json
import logging
import os
import time
from datetime import datetime
import gc
import comet_ml
import pickle
import pandas as pd

import numpy as np
import optuna
from optuna.samplers import TPESampler
import lightning as L
import torch
import GPUtil
import threading
from threading import Thread
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import CometLogger, CSVLogger
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, DataLoader, Subset
from torch_geometric.data import DataLoader as GraphDataLoader
import psutil  # For CPU memory tracking
import multiprocessing
from sklearn.model_selection import GroupShuffleSplit

from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from source.pytorch.datasets import (
    MiTarDataset, CustomDataset, PredictionDataset, HelwakDataset,
    MiTarGraphDataset, CustomGraphDataset, HomologyDataset
)
from source.pytorch.modules import MiTarModule

CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/")
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

######## Utility Functions #######
def get_max_memory_usage():
    """Returns the maximum memory usage in MB, using CUDA if available, else CPU."""
    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        torch.cuda.reset_max_memory_allocated()
        return max_memory
    else:
        process = psutil.Process(os.getpid())
        max_memory = process.memory_info().rss / (1024 ** 2)
        return max_memory

def save_paths(Paths_list, file_name):
    

    # Example list of Path objects
    path_list = Paths_list
    
    # File path to save the paths
    if ".pkl" not in file_name:
        file_name = file_name + ".pkl"

    # Serialize and save the list of Path objects
    with open(file_name, "wb") as file:
        pickle.dump(path_list, file)
    
    print(f'Saved file {file_name}')



def load_paths(file_name):
    if ".pkl" not in file_name:
        file_name = file_name + ".pkl"
    with open(file_name, "rb") as file:
        imported_path_list = pickle.load(file)
    print(f'Opened file {file_name}')
    return imported_path_list





'''
gpu_max_utilization = 0  # Initialize to track maximum utilization
total_utilization = 0  # Initialize to track total utilization
num_samples = 0  # Count the number of samples

def log_gpu_utilization(interval=10):
    """Log maximum and average GPU utilization every `interval` seconds."""
    global logging_enabled, gpu_max_utilization, total_utilization, num_samples  # Declare the variables as global

    while logging_enabled:
        # Get the list of available GPUs
        gpus = GPUtil.getGPUs()

        print(f"gpus are {gpus}")

        # Check each GPU's utilization
        for gpu in gpus:
            utilization = gpu.load * 100  # Convert to percentage
            
            # Update max utilization
            if utilization > gpu_max_utilization:
                gpu_max_utilization = utilization
            
            # Update total utilization and sample count
            total_utilization += utilization
            num_samples += 1

        # Wait for the specified interval
        time.sleep(interval)

# Function to start logging
def start_logging():
    global logging_enabled
    logging_enabled = True
    logging_thread = threading.Thread(target=log_gpu_utilization, daemon=True)
    logging_thread.start()  # Start the logging function

# Function to stop logging
def stop_logging():
    global logging_enabled
    logging_enabled = False
        
'''

class Monitor(Thread): # Changed here ##################
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay  # Time between calls to GPUtil
        self.gpu_utilization_values = []  # Store GPU usage values
        self.start()

    def run(self):
        while not self.stopped:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usage = gpus[0].load * 100  # Convert to percentage
                self.gpu_utilization_values.append(gpu_usage)
                print(f"GPU Utilization: {gpu_usage}%")
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
        self.join()  # Ensure the thread stops before calculating stats

    def get_max_utilization(self):
        return max(self.gpu_utilization_values) if self.gpu_utilization_values else 0

    def get_avg_utilization(self):
        return sum(self.gpu_utilization_values) / len(self.gpu_utilization_values) if self.gpu_utilization_values else 0

    

def log_pytorch_gpu_usage(): # Changed here ##################
    gpu_id = 0
    gpu_memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**2
    print(f"Memory Allocated: {gpu_memory_allocated:.2f} MB")
    gpu_memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**2
    print(f"Memory Reserved: {gpu_memory_reserved:.2f} MB")
    return gpu_memory_allocated, gpu_memory_reserved


######## Main Function #######
def main(args):
    overall_start = time.time()
    L.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ######## Setup Logging Directories #######
    # Create directories for logs and trial results
    log_root = args.log_dir  # e.g., 'tecmitarget_logs'
    log_root = f"{args.dataset_type}_{log_root}"
    objective_log_dir = os.path.join(log_root, "objective")  # e.g., 'tecmitarget_logs/objective'
    os.makedirs(log_root, exist_ok=True)
    os.makedirs(objective_log_dir, exist_ok=True)

    test_reg_directory = os.path.join(log_root, "Temp_Test_Registeration")

    ######## Dataset Loading #######
    # Load the appropriate dataset based on args.dataset
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
    elif args.dataset == "Homology": # Changed here ####################
            
        # Load the fold from pickle
        train_data = load_paths(os.path.join(args.input_data_path ,"train_data.pkl"))
        test_data = load_paths(os.path.join(args.input_data_path ,"test_data.pkl"))
        #with open(os.path.join(args.input_data_path.lower() ,"train_data.pkl"), "rb") as f:
        #    train_data = pickle.load(f)
        # Load the fold from pickle
        #with open(os.path.join(args.input_data_path.lower() ,"test_data.pkl"), "rb") as f:
        #    test_data = pickle.load(f)
        
        
    elif args.dataset == "Helwak":
        dataset = HelwakDataset(
            args.input_data_path,
            strip_n=args.strip_n,
            n_encoding=tuple([float(n.strip()) for n in args.n_encoding.split(",")]),
            icshape=args.icshape
        )
    else:
        raise ValueError(f"Invalid dataset name {args.dataset}")

    # Set the appropriate DataLoader type (standard or graph-based)
    dataloader = DataLoader
    if args.model == "GraphTar":
        dataloader = GraphDataLoader

    ######## Model Configuration #######
    # Define the TEC-miTarget model configuration
    if args.model == "TEC-miTarget":
        if args.dataset=="Homology": # Changed here ###################
            model_config = {
                #'max_mirna_len': dataset.max_mirna_len,
                #'max_target_len': dataset.max_target_len,
                'input_dim': 512, 'projection_dim': 256, 'n_heads': 1,
                'n_layers': 6, 'dropout': 0, 'kernal_size': 9,
                'p0': 0.5, 'gamma_init': 0
            }
        else:
            model_config = {
                    'max_mirna_len': train_dataset.max_mirna_len,
                    'max_target_len': train_dataset.max_target_len,
                    'input_dim': 512, 'projection_dim': 256, 'n_heads': 1,
                    'n_layers': 6, 'dropout': 0, 'kernal_size': 9,
                    'p0': 0.5, 'gamma_init': 0
                }
    else:
        raise ValueError(f"Invalid model name {args.model}")

    

    ######## Cross-Validation Setup #######
    # Initialize Stratified K-Fold for balanced splits
    skf = StratifiedKFold(n_splits=args.num_cv_folds, shuffle=True, random_state=args.seed)
    all_fold_results = {}
    exported_splits = {}
    process = psutil.Process()  # For monitoring system resources

    

    ######## Fold Loop #######
    folds_list = []
    train_val_list = []
    test_list = []
    if args.dataset != "Homology": # Changed here ###################
        ######## Label Extraction for Stratified Split #######
        # Extract labels for stratified splitting to maintain class balance
        try:
            labels = [sample[2] for sample in dataset]
        except Exception as e:
            log.error("Error extracting labels: %s", e)
            labels = [int(data.y) for data in dataset]
        for fold, (train_val_indices, test_indices) in enumerate(skf.split(dataset, labels)): # Changed here Check if random was here
            folds_list.append(fold)
            train_val_list.append(train_val_indices)
            test_list.append(test_indices)
            
    else:
        
        for fold in train_data.keys():
            folds_list.append(int(fold))
            train_val_list.append(train_data[str(fold)])
            test_list.append(test_data[str(fold)])


    ##### Describe a variable for_folds # Changed here ##############
    #If not Homology:
    #    do skf splitting (store in for_folds)
    #else:
    #    for_folds = zip(folds_list, train_val_list, test_list)

    for fold, train_val_indices, test_indices in zip(folds_list, train_val_list, test_list): # Changed here ############## Not changed just check
        #if fold > 3:
        #    ("Fold is greater than the required fold. Not running the remaining folds.")
        #    continue
        fold_start = time.time()
        log.info(f"Starting fold {fold}")

        if args.dataset == "Homology":
            df_fold = train_val_indices.copy()
            df_fold_H_neg = df_fold[df_fold["Homology"]=="H_neg"]
            df_fold_H_pos = df_fold[df_fold["Homology"]=="H_pos"]
            df_fold_NH_neg = df_fold[df_fold["Homology"]=="NH_neg"]
            df_fold_NH_pos = df_fold[df_fold["Homology"]=="NH_pos"]
            df_fold_NH_neg_Train, df_fold_NH_neg_Val = train_test_split(df_fold_NH_neg, test_size = args.val_ratio*2, random_state = args.seed)
            df_fold_NH_pos_Train, df_fold_NH_pos_Val = train_test_split(df_fold_NH_pos, test_size = args.val_ratio*2, random_state = args.seed)
            train_df = pd.concat([df_fold_H_neg, df_fold_NH_neg_Train, df_fold_H_pos, df_fold_NH_pos_Train]).reset_index(drop=True)
            val_df =  pd.concat([df_fold_NH_neg_Val, df_fold_NH_pos_Val]).reset_index(drop=True)
            
            # Drop homology
            train_df.drop(columns=["Homology"], inplace=True)
            val_df.drop(columns=["Homology"], inplace=True)
            test_df = test_indices.copy()
            # Wrap into datasets
            train_dataset = HomologyDataset(input_dataset=train_df,
                n_encoding=tuple([float(n.strip()) for n in args.n_encoding.split(",")]))
            val_dataset = HomologyDataset(input_dataset=val_df,
                n_encoding=tuple([float(n.strip()) for n in args.n_encoding.split(",")]))
            test_dataset = HomologyDataset(input_dataset=test_df,
                n_encoding=tuple([float(n.strip()) for n in args.n_encoding.split(",")]))
        else:
            
            # Convert indices to numpy arrays for easier manipulation
            train_val_indices = np.array(train_val_indices)
            test_indices = np.array(test_indices)
    
            # Split train_val into training and validation sets
            rng = np.random.RandomState(args.seed + fold)
            total_train_val = len(train_val_indices)
            val_size = int(args.val_ratio * total_train_val)
            shuffled = train_val_indices.copy()
            rng.shuffle(shuffled)
            val_indices = shuffled[:val_size]
            train_indices = shuffled[val_size:]
    
            # Save split indices for reproducibility
            exported_splits[f"fold_{fold}"] = {
                "train_indices": train_indices.tolist(),
                "val_indices": val_indices.tolist(),
                "test_indices": test_indices.tolist()
            }
    
            train_val_set = Subset(dataset, train_val_indices.tolist())

        #if args.model == "TEC-miTarget":
        #    if args.dataset=="Homology":
        #        model_config = {
        #            'max_mirna_len': train_dataset.max_mirna_len,
        #            'max_target_len': train_dataset.max_target_len,
        #            'input_dim': 512, 'projection_dim': 256, 'n_heads': 1,
        #            'n_layers': 6, 'dropout': 0, 'kernal_size': 9,
        #            'p0': 0.5, 'gamma_init': 0
        #        }
        trial_results = []

        ######## Trial Training #######
        # Define the objective function for Optuna hyperparameter optimization
        def objective(trial):
            # Sample hyperparameters to optimize
            opt_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'Adadelta'])
            dropout_rate = trial.suggest_categorical('dropout_rate', [0.1, 0.2, 0.3, 0.4, 0.5])
            trial_batch_size = trial.suggest_categorical('batch_size', [10, 30, 50, 100, 200])
            if args.dataset != "Homology":
                total = len(train_val_set)
                val_size_trial = int(0.1 * total)
                train_size_trial = total - val_size_trial
                train_subset_trial, val_subset_trial = random_split(train_val_set, [train_size_trial, val_size_trial])
                # Create data loaders for the trial
                train_loader = dataloader(train_subset_trial, batch_size=trial_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
                val_loader = dataloader(val_subset_trial, batch_size=trial_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            else: # Changed here ####################
                df_fold = train_val_indices.copy()
                df_fold_H_neg = df_fold[df_fold["Homology"]=="H_neg"]
                df_fold_H_pos = df_fold[df_fold["Homology"]=="H_pos"]
                df_fold_NH_neg = df_fold[df_fold["Homology"]=="NH_neg"]
                df_fold_NH_pos = df_fold[df_fold["Homology"]=="NH_pos"]
                df_fold_NH_neg_Train, df_fold_NH_neg_Val = train_test_split(df_fold_NH_neg, test_size = 0.2)
                df_fold_NH_pos_Train, df_fold_NH_pos_Val = train_test_split(df_fold_NH_pos, test_size = 0.2)
                train_df = pd.concat([df_fold_H_neg, df_fold_NH_neg_Train, df_fold_H_pos, df_fold_NH_pos_Train]).reset_index(drop=True)
                val_df =  pd.concat([df_fold_NH_neg_Val, df_fold_NH_pos_Val]).reset_index(drop=True)
                
                #train_df = df_fold.iloc[train_idx].reset_index(drop=True)
                #val_df = df_fold.iloc[val_idx].reset_index(drop=True)
                # Drop Homology
                train_df.drop(columns=["Homology"], inplace=True)
                val_df.drop(columns=["Homology"], inplace=True)
                # Wrap into datasets
                train_subset_trial = HomologyDataset(input_dataset=train_df,
                    n_encoding=tuple([float(n.strip()) for n in args.n_encoding.split(",")]))
                val_subset_trial = HomologyDataset(input_dataset=val_df,
                    n_encoding=tuple([float(n.strip()) for n in args.n_encoding.split(",")]))
                train_loader = dataloader(train_subset_trial, batch_size=trial_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
                val_loader = dataloader(val_subset_trial, batch_size=trial_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

            

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

            log.info(f"Fold {fold}, Trial {trial.number} - Hyperparameters: optimizer={opt_name}, dropout_rate={dropout_rate}, batch_size={trial_batch_size}, opt_config={opt_config}")

            if torch.cuda.is_available():
                torch.cuda.reset_max_memory_allocated()

            # Variables to store maximum and average utilization
            #gpu_max_utilization = 0  # Initialize to track maximum utilization
            #total_utilization = 0  # Initialize to track total utilization
            #num_samples = 0  # Count the number of samples
            #start_logging() # For GPU Utilisation
            monitor = Monitor(10) # Chenged here ######## Just check


            

            # Adjust model config with trial-specific dropout
            current_model_config = model_config.copy()
            current_model_config['dropout'] = dropout_rate

            # Initialize the model
            model = MiTarModule(
                model_name=args.model,
                model_config=current_model_config,
                opt_name=opt_name,
                opt_config=opt_config
            )

            # Setup callbacks and trainer for the trial
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
                default_root_dir=os.path.join(CHECKPOINT_PATH, f"{args.model}/{args.dataset_type}/fold{fold}/trial_{trial.number}"), # Cahanged here #### Included dataset type
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

            # Monitor resources during training
            start_time = time.time()
            mem_before = process.memory_info().rss / (1024 ** 2)
            log.info(f"Fold {fold}, Trial {trial.number} - Memory before training: {mem_before:.2f} MB")

            trainer.fit(model, train_loader, val_loader)

            end_time = time.time()
            time_taken = end_time - start_time
            mem_after = process.memory_info().rss / (1024 ** 2)
            cpu_percent = psutil.cpu_percent(interval=None)
            max_memory_mb = get_max_memory_usage()
            gpu_memory_allocated, gpu_memory_reserved = log_pytorch_gpu_usage()
            #stop_logging()
            # Close monitor
            monitor.stop()

            gpu_average_utilization = monitor.get_avg_utilization()
            gpu_max_utilization = monitor.get_max_utilization()
            

            log.info(f"Fold {fold}, Trial {trial.number} - Training time: {time_taken:.2f} sec")
            log.info(f"Fold {fold}, Trial {trial.number} - Memory after training: {mem_after:.2f} MB")
            log.info(f"Fold {fold}, Trial {trial.number} - Peak CPU usage: {cpu_percent:.2f}%")
            log.info(f"Fold {fold}, Trial {trial.number} - Max memory usage: {max_memory_mb:.2f} MB")
            log.info(f"Fold {fold}, Trial {trial.number} - GPU Average Utilization: {gpu_average_utilization:.2f}%")
            log.info(f"Fold {fold}, Trial {trial.number} - GPU Max Utilization: {gpu_max_utilization:.2f}%")
            log.info(f"Fold {fold}, Trial {trial.number} - GPU Memory Allocated: {gpu_memory_allocated:.2f} MB")
            log.info(f"Fold {fold}, Trial {trial.number} - GPU Memory Reserved: {gpu_memory_reserved:.2f} MB")

            # Evaluate and store trial results
            trial_value = checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score is not None else float('-inf')
            val_metrics = trainer.validate(model, val_loader)[0] if trainer.validate(model, val_loader) else {}
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
                "gpu_average_utilization": gpu_average_utilization,
                "gpu_max_utilization": gpu_max_utilization,
                "gpu_memory_allocated": gpu_memory_allocated,
                "gpu_memory_reserved": gpu_memory_reserved,
                "timestamp": datetime.now().isoformat(),
                "best_model_path": checkpoint_callback.best_model_path,
                "training_time_sec": time_taken,
                "cpu_percent": cpu_percent,
                "memory_before_mb": mem_before,
                "memory_after_mb": mem_after
            }
            #trial_results.append(trial_info)

            with open(trial_results_temp_file, "a") as f:
                json.dump(trial_info, f)
                f.write("\n")  # Add newline to separate JSON objects

            
                        # Clear memory after trial
            del model, trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return trial_value

        ######## Hyperparameter Optimization #######
        # Number of workers
        number_of_workers = args.num_workers * 2
        number_of_workers_to_use = args.num_workers

        print(f"Using {number_of_workers_to_use} workers out of {number_of_workers} workers available.")
        
        # Run Optuna optimization to find the best hyperparameters
        trial_results_temp_file = os.path.join(log_root, f"Temp_trial_results_tecmitarget_{fold}.jsonl")
        os.makedirs(f"{log_root}/optuna", exist_ok=True)
        n_trials=args.num_trials
        study_name = f"Study_fold_{fold}"
        storage=f'sqlite:///{log_root}/optuna/{study_name}.db'
        study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize", sampler=TPESampler(seed=args.seed + fold), load_if_exists=True)
        
        #study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=args.seed + fold))
        #study.optimize(objective, n_trials=n_trials)
        
        # Ensure the study contains trials before filtering
        if len(study.trials) > 0:
            # Count completed trials safely
            completed_trials = sum(1 for trial in study.trials if trial.state == TrialState.COMPLETE)
            # Count active (running) trials
            running_trials = sum(1 for trial in study.trials if trial.state == TrialState.RUNNING)
        else:
            completed_trials = 0  # No trials exist yet
            running_trials = 0
        
        # Check if 4 trials are already running
        if running_trials >= args.gpus_to_use: #Trials per study
            print(f"⏳ Already {running_trials} trials running. Skipping...")
            continue
        else:
            # Ensure we only run the remaining required trials
            remaining_trials = max(0, n_trials - completed_trials)

            if remaining_trials > 0:
                print(f"Running {remaining_trials} more trials...")
                study.optimize(objective, callbacks=[MaxTrialsCallback(n_trials, states=(TrialState.COMPLETE,))])
            else:
                print(f"✅ Required {n_trials} successful trials already completed. Skipping optimization.")
                
                
        best_params = study.best_trial.params
        log.info(f"Best params for fold {fold}: {best_params}")



        # Check if all the trials are already completed.
        # Else, skip the test and when all the trials will be completed the last run which completes the last trial will perform the test

        
        # Count completed trials safely
        completed_trials_post_optimisation = sum(1 for trial in study.trials if trial.state == TrialState.COMPLETE)
        if completed_trials_post_optimisation >= n_trials:
            print("All trials completed. Checking if the test hasn't been run earlier.")
        else:
            print("All trials not complete. Skipping the test for now. Waiting for all the trials to complete.")
            continue


        # Loading trial results from temporary file

        trial_results = []
        with open(trial_results_temp_file, "r") as f:
            for line in f:
                try:
                    trial_results.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid line: {line.strip()} (Error: {e})")

        best_trial_metrics = next((t["validation_metrics"] for t in trial_results if t["trial_number"] == study.best_trial.number), {})

        fold_summary = {
            "trial_results": trial_results,
            "best_trial": {
                "trial_number": study.best_trial.number,
                "hyperparameters": best_params,
                "best_value": study.best_trial.value,
                "validation_metrics": best_trial_metrics
            }
        }

        ######## Final Model Preparation #######
        
        
        # We want to log everything only once and run the test once
        # Hence, we create a dummy file to see if it exists then we don't perform the test again.
        

        if not os.path.exists(test_reg_directory):
            try:
                os.makedirs(test_reg_directory)
                print(f"Directory '{test_reg_directory}' created successfully.")
            except PermissionError:
                print(f"Permission denied: Unable to create '{test_reg_directory}'. Try running with sudo or changing permissions.\nTest not run. Exiting...")
                continue

        else:
            print(f"Directory '{test_reg_directory}' already exists.")

        if not os.path.exists(test_reg_directory):
            try:
                os.makedirs(test_reg_directory)
                print(f"Directory '{test_reg_directory}' created successfully.")
            except PermissionError:
                print(f"Permission denied: Unable to create '{test_reg_directory}'. Try running with sudo or changing permissions.\nTest not run. Exiting...")
                continue

        else:
            print(f"Directory '{test_reg_directory}' already exists.")

        current_test_flag_file = f'{test_reg_directory}/Test_fold_{fold}'

        if not os.path.exists(current_test_flag_file):
            print("Running the test...")
            # Mark the test as done
            with open(current_test_flag_file, "w") as f:
                f.write("Test Run.")
        else:
            print("Test has already been run. Skipping...")
            continue
        
        # Prepare datasets and loaders for final training and testing
        if args.dataset!="Homology":
            
            current_splits = exported_splits[f"fold_{fold}"]
            train_subset = Subset(dataset, current_splits["train_indices"])
            val_subset = Subset(dataset, current_splits["val_indices"])
            test_subset = Subset(dataset, current_splits["test_indices"])
    
            final_batch_size = best_params["batch_size"]
            train_loader = dataloader(train_subset, batch_size=final_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            val_loader = dataloader(val_subset, batch_size=final_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            test_loader = dataloader(test_subset, batch_size=64, shuffle=False, num_workers=args.num_workers, pin_memory=False)
        else:
            final_batch_size = best_params["batch_size"]
            train_loader = dataloader(train_dataset, batch_size=final_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            val_loader = dataloader(val_dataset, batch_size=final_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            test_loader = dataloader(test_dataset, batch_size=64, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            
        # Configure the final model with best hyperparameters
        final_model_config = model_config.copy()
        final_model_config['dropout'] = best_params['dropout_rate']

        final_opt_name = best_params['optimizer']
        final_opt_config = {k: v for k, v in best_params.items() if k.startswith('lr_') or k in ['momentum', 'rho']}
        if 'lr_' + final_opt_name.lower() in final_opt_config:
            final_opt_config['lr'] = final_opt_config.pop('lr_' + final_opt_name.lower())
        final_opt_config['weight_decay'] = args.weight_decay

        log.info(f"Fold {fold} - Final model hyperparameters: optimizer={final_opt_name}, dropout_rate={best_params['dropout_rate']}, batch_size={final_batch_size}, opt_config={final_opt_config}")

        # Load the best trial's weights
        best_model_path = next((t["best_model_path"] for t in trial_results if t["trial_number"] == study.best_trial.number), None)
        if not best_model_path:
            raise ValueError("Best model path not found")

        # Delete the temporary trial results file

        #if os.path.exists(trial_results_temp_file):
        #    os.remove(trial_results_temp_file)
        #    print(f"File '{trial_results_temp_file}' deleted successfully!")
        #else:
        #    print(f"File '{trial_results_temp_file}' does not exist.")

        # Continue with the trial results

        final_model = MiTarModule(
            model_name=args.model,
            model_config=final_model_config,
            opt_name=final_opt_name,
            opt_config=final_opt_config
        )
        checkpoint = torch.load(best_model_path)
        final_model.load_state_dict(checkpoint["state_dict"])
        # Setup final trainer
        save_name = f"{args.model}/fold-{fold}"
        final_logger = CometLogger(
            api_key="7UP5qmq9VeL4BBoYLs4ymMKpA", #"UVSOKGVLOrkjjS58XIIfdKPFN",
            project_name=f"dlmirtpred-v1_{args.dataset_type}_final_run_fold_{fold}",
            workspace="sarmadak"
        ) if args.comet_logging else CSVLogger(save_dir=log_root, name=save_name)
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

        ######## Final Training #######
        # Train the final model with the best weights
        log.info(f"Training final model for fold {fold}...")
        start_time = time.time()
        mem_before = process.memory_info().rss / (1024 ** 2)
        log.info(f"Fold {fold} - Memory before final training: {mem_before:.2f} MB")
        # Variables to store maximum and average utilization
        #gpu_max_utilization = 0  # Initialize to track maximum utilization
        #total_utilization = 0  # Initialize to track total utilization
        #num_samples = 0  # Count the number of samples
        #start_logging() # For GPU Utilisation
        monitor = Monitor(10)

        if torch.cuda.is_available():
            torch.cuda.reset_max_memory_allocated()
        final_trainer.fit(final_model, train_loader, val_loader)

        end_time = time.time()
        time_taken = end_time - start_time
        mem_after = process.memory_info().rss / (1024 ** 2)
        cpu_percent = psutil.cpu_percent(interval=None)
        max_memory_mb = get_max_memory_usage()
        gpu_memory_allocated, gpu_memory_reserved = log_pytorch_gpu_usage()
        #stop_logging()
        monitor.stop()

        gpu_average_utilization = monitor.get_avg_utilization()
        gpu_max_utilization = monitor.get_max_utilization()
        # Calculate average utilization
        #gpu_average_utilization = total_utilization / num_samples if num_samples > 0 else 0
        

        log.info(f"Fold {fold} - Final training time: {time_taken:.2f} sec")
        log.info(f"Fold {fold} - Memory after final training: {mem_after:.2f} MB")
        log.info(f"Fold {fold} - Peak CPU usage: {cpu_percent:.2f}%")
        log.info(f"Fold {fold} - Max memory usage: {max_memory_mb:.2f} MB")
        log.info(f"Fold {fold} - GPU Average Utilization: {gpu_average_utilization:.2f}%")
        log.info(f"Fold {fold} - GPU Max Utilization: {gpu_max_utilization:.2f}%")
        log.info(f"Fold {fold} - GPU Memory Allocated: {gpu_memory_allocated:.2f} MB")
        log.info(f"Fold {fold} - GPU Memory Reserved: {gpu_memory_reserved:.2f} MB")

        ######## Testing on CPU #######
        # Test the final model on CPU to manage memory
        log.info(f"Switching not to CPU for testing fold {fold}...") ### Remove not to return to default
        #final_model = final_model.cpu()
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Changed here #########
        final_model.to(device)
        test_trainer = L.Trainer(
            accelerator='gpu',#'auto',
            devices=1,
            inference_mode=True
        )
        
        log.info(f"Testing final model for fold {fold} on GPU, not CPU...")
        start_time = time.time()
        mem_before_test = process.memory_info().rss / (1024 ** 2)
        log.info(f"Fold {fold} - Memory before testing: {mem_before_test:.2f} MB")
        # Variables to store maximum and average utilization
        #gpu_max_utilization = 0  # Initialize to track maximum utilization
        #total_utilization = 0  # Initialize to track total utilization
        #num_samples = 0  # Count the number of samples
        #start_logging() # For GPU Utilisation
        monitor = Monitor(10)

        del train_loader, val_loader  # Free memory
        #test_results = test_trainer.test(final_model, test_loader)[0] if test_trainer.test(final_model, test_loader) else {}
        
        test_results = test_trainer.test(final_model, test_loader)
        test_results = test_results[0] if test_results else {}

        end_time = time.time()
        time_taken_test = end_time - start_time
        mem_after_test = process.memory_info().rss / (1024 ** 2)
        cpu_percent_test = psutil.cpu_percent(interval=None)
        max_memory_mb_test = get_max_memory_usage()
        gpu_memory_allocated, gpu_memory_reserved = log_pytorch_gpu_usage()
        #stop_logging()
        monitor.stop()

        gpu_average_utilization = monitor.get_avg_utilization()
        gpu_max_utilization = monitor.get_max_utilization()
        # Calculate average utilization
        #gpu_average_utilization = total_utilization / num_samples if num_samples > 0 else 0

        log.info(f"Fold {fold} - Testing time: {time_taken_test:.2f} sec")
        log.info(f"Fold {fold} - Memory after testing: {mem_after_test:.2f} MB")
        log.info(f"Fold {fold} - Peak CPU usage during testing: {cpu_percent_test:.2f}%")
        log.info(f"Fold {fold} - Max memory usage during testing: {max_memory_mb_test:.2f} MB")
        log.info(f"Fold {fold} - GPU Average Utilization during testing: {gpu_average_utilization:.2f}%")
        log.info(f"Fold {fold} - GPU Max Utilization during testing: {gpu_max_utilization:.2f}%")
        log.info(f"Fold {fold} - GPU Memory Allocated during testing: {gpu_memory_allocated:.2f} MB")
        log.info(f"Fold {fold} - GPU Memory Reserved during testing: {gpu_memory_reserved:.2f} MB")

        ######## Prediction (Optional) #######
        # Perform predictions if a prediction dataset is provided
        if args.prediction_data_path:
            log.info(f"Predicting for fold {fold} on GPU, not CPU...")
            prediction_dataset = PredictionDataset(
                args.prediction_data_path,
                max_mirna_len=dataset.max_mirna_len,
                max_target_len=dataset.max_target_len,
                n_encoding=dataset.n_encoding,
                strip_n=args.strip_n
            )
            prediction_loader = DataLoader(prediction_dataset, batch_size=64, shuffle=False, num_workers=args.num_workers, pin_memory=False)
            predictions = test_trainer.predict(final_model, prediction_loader)
            fold_summary["predictions"] = [p.tolist() for p in predictions]

        ######### Perform prediction and probabilities extraction ###############
        '''
        all_labels = torch.cat([x["labels"] for x in predictions], dim=0)
        all_preds = torch.cat([x["preds"] for x in predictions], dim=0)
        all_probs = torch.cat([x["probs"] for x in predictions], dim=0)

        # Convert to NumPy
        all_labels_np = all_labels.cpu().numpy()
        all_preds_np = all_preds.cpu().numpy()
        all_probs_np = all_probs.cpu().numpy()
        '''
        '''

        final_model.eval()  # Set model to evaluation mode
        all_preds = []
        all_probs = []

        with torch.no_grad():  # Disable gradient computation
            for batch in test_loader:
                inputs, labels = batch  # Adjust if dataset has different format
                outputs = final_model(inputs)  # Forward pass

                probs = torch.softmax(outputs, dim=1)  # Convert logits to probabilities (for multi-class)
                preds = torch.argmax(probs, dim=1)  # Get predicted class

                all_preds.extend(preds.cpu().numpy())  # Store predictions
                all_probs.extend(probs.cpu().numpy())  # Store probability values

        '''
        '''
        # Convert predictions to class labels and probabilities
        for batch in predictions:
            probs = torch.softmax(batch, dim=1)  # Convert logits to probabilities
            preds = torch.argmax(probs, dim=1)  # Get predicted class

            all_preds.extend(preds.cpu().numpy())  # Store predictions
            all_probs.extend(probs.cpu().numpy())  # Store probability values
        
        # Extract predictions and probabilities
        all_preds = torch.cat([x["preds"] for x in test_results]).cpu().numpy()
        all_probs = torch.cat([x["probs"] for x in test_results]).cpu().numpy()
        '''
        '''
        # Save predictions and probabilities
        np.save(os.path.join(objective_log_dir, f"test_predictions_fold_{fold}.npy"), np.array(all_preds_np))
        np.save(os.path.join(objective_log_dir, f"test_probabilities_fold_{fold}.npy"), np.array(all_probs_np))
        np.save(os.path.join(objective_log_dir, f"test_labels_fold_{fold}.npy"), np.array(all_labels_np))
        '''

        ######## Fold Summary #######
        # Compile and save fold results
        fold_summary["final_metrics"] = test_results
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
        fold_summary["final_gpu_average_utilization"] = gpu_average_utilization
        fold_summary["final_gpu_max_utilization"] = gpu_max_utilization
        fold_summary["final_gpu_memory_allocated_mb"] = gpu_memory_allocated
        fold_summary["final_gpu_memory_reserved_mb"] = gpu_memory_reserved
        fold_summary["final_hyperparameters"] = {
            "optimizer": final_opt_name,
            "dropout_rate": best_params['dropout_rate'],
            "batch_size": final_batch_size,
            **final_opt_config
        }

        fold_end = time.time()
        fold_summary["fold_execution_time_sec"] = fold_end - fold_start
        log.info(f"Fold {fold} executed in {fold_summary['fold_execution_time_sec']:.2f} seconds.")

        fold_results_file = os.path.join(objective_log_dir, f"fold{fold}_results.json")
        with open(fold_results_file, "w") as f:
            json.dump(fold_summary, f, indent=4)
        all_fold_results[f"fold_{fold}"] = fold_summary
        # Clear memory after fold
        del final_model, final_trainer, test_trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ######## Final Output #######
    # Save all results and splits
    with open(os.path.join(log_root, "all_folds_summary.json"), "w") as f:
        json.dump(all_fold_results, f, indent=4)
    with open(os.path.join(log_root, "exported_splits.json"), "w") as f:
        json.dump(exported_splits, f, indent=4)
    overall_time = time.time() - overall_start
    log.info(f"Total execution time: {overall_time:.2f} seconds.")

    # See if the Test log folder exists

    if not os.path.exists(test_reg_directory):
        print("Tests were not run due to the error in the test folder creation")
## CLIPL Dataset Name training_set_intarna_filtered_no_slop_CLIPL.tsv
######## Command Line Interface #######
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=500, help='Maximum number of epochs to train for')                    # change here
    parser.add_argument('--batch-size', type=int, default=32, help='Default batch size (overridden by Optuna)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default, tuned via Optuna)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=1, help='Early stopping patience')
    parser.add_argument('--num-cv-folds', type=int, default=5, help='Number of cross-validation folds')                     # change here
    parser.add_argument('--num-trials', type=int, default=100, help='Number of Optuna trials per fold')                       # change here
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--model', type=str, default='TEC-miTarget',
                        choices=['CNNSequenceModel', 'DMISO', 'MiTar', 'TransPHLA', 'TEC-miTarget', 'GraphTar'],
                        help='Model name')
    parser.add_argument('--comet-logging', action='store_true', help='Enable Comet.ml logging')
    parser.add_argument('--dataset', type=str, default='Homology', choices=['MiTar', 'Custom', 'Helwak', 'Homology'], help='Dataset')
    parser.add_argument('--input-data-path', type=str,
                        default='data/homology/ClipL', help='Input data path')
    parser.add_argument('--prediction-data-path', type=str, default=None, help='Prediction data path')
    parser.add_argument('--canonical-only', action='store_true', help='Use only canonical pairs')
    parser.add_argument('--incorporate-type', action='store_true', help='Incorporate interaction type')
    parser.add_argument('--strip-n', action='store_true', help='Strip padding Ns')
    parser.add_argument('--n-encoding', type=str, default='1,0,0,0,0', help='N encoding')
    parser.add_argument('--icshape', action='store_true', help='Use icSHAPE data')
    parser.add_argument('--word2vec-model-dir', type=str, default='data/mitar/word2vec_models', help='Word2Vec models directory')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data loader workers') # int(max(1, multiprocessing.cpu_count() // 2))
    parser.add_argument('--log-dir', type=str, default='tecmitarget_logs', help='Log directory')
    parser.add_argument('--dataset-type', type=str, default='Homology_ClipL', help='Optional Dataset Type Name')
    parser.add_argument('--gpus-to-use', type=int, default=1, help='Number of GPUs to use')

    args = parser.parse_args()
    print("Arguments: ", args)
    main(args)
