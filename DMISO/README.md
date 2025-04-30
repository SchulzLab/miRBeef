## `train_cv_dmiso.py`

A PyTorch-Lightning + Optuna pipeline for nested cross-validation and hyperparameter optimization of the **DMISO** model on miRNA–target interaction data.

---

### 🔑 Features

- **Nested Stratified K-Fold CV**  
  Configurable `--num-cv-folds` and inner `--val-ratio`.  
- **Optuna TPE sampler** over:
  - Optimizer: `Adam`, `SGD`, `Adadelta`  
  - Learning rates & momentum/rho  
  - Dropout rates  
  - Batch sizes  
- **Early Stopping** on validation accuracy  
- **Resource Logging**: timing, CPU/GPU memory usage, CPU%  
- **Comet.ml** support (`--comet-logging`) or local CSV logs  
- **Final Retrain & Test**: best trial → full train+val → held-out test

---

### ⚙️ Installation

```bash
git clone git@github.com:SchulzLab/miRBeef.git
cd DMISO
pip install -r requirements.txt
```


### 🚀 Usage
```bash
#### Set the working directory

export REPO_DIR=$(pwd)
python train_cv_dmiso.py \
  --comet-logging \
  --model DMISO \
  --dataset MiTar \
  --input-data-path $REPO_DIR/data/mitar/allCLIP_final.txt \
  --strip-n
```



### 📂 Outputs
- **Checkpoints:
saved_models/DMISO/fold-{i}/…

- **Logs:

Comet.ml dashboard (if enabled)

Local CSVs under miTar_log_DMISO/...

- **Summaries:

miTar_log_DMISO/all_folds_summary.json

miTar_log_DMISO/exported_splits.json

Per-fold JSON: miTar_log_DMISO/objective/fold{i}_results.json

**See the code comments in train_cv_dmiso.py for the full list of command-line options and defaults.







