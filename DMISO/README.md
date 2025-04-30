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


### 🚀 Usage
```
export REPO_DIR=$(pwd)
python train_cv_dmiso.py \
  --comet-logging \
  --model DMISO \
  --dataset MiTar \
  --input-data-path $REPO_DIR/data/mitar/allCLIP_final.txt \
  --strip-n
```