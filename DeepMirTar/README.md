# DeepMirTar v2

DeepMirTar v2 combines miRanda alignments with a deep learning model to predict miRNA–mRNA interactions efficiently and accurately.

---

### ⚙️ Installation
#### Ensure the miranda executable is on your PATH
```bash
git clone git@github.com:SchulzLab/miRBeef.git
cd DeepMirTar
conda create -n deepmirtar python=3.8 -y
conda activate deepmirtar
pip install -r requirements.txt

```


### 🚀 Usage
```bash
#### Set the working directory
python DeepMirTar_modified-final.py \
  -c /path/to/input.txt \
  -o /path/to/output.tspr \
  -t 0.5 \
  --metrics performance_metrics.tsv

```
- **-c / --csv: Input TSV/txt with columns:**
miRNA_id, miRNA_seq, target_id, target_seq[, label].

- **-o / --output: Output .tspr file containing predicted sites.**

- **-t / --threshold: Score cutoff for calling a positive site (default: 0.5).**

- **--metrics: (Optional) Path to save a TSV of performance metrics.**


### 📂 Outputs

- **Logs**:

Console & File: All logs (info, warnings, errors) go to both STDOUT and deepmirtar.log









