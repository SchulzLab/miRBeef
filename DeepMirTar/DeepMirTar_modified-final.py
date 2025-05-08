#!/usr/bin/env python
"""
DeepMirTar v2: miRNA–mRNA target prediction with optional performance metrics and progress logging.
Errors (e.g. failed feature extraction) are logged to both console and 'deepmirtar.log'.
"""
import argparse
import logging
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from RNA_features_site_level import get_all_RNA_site_level_features
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def standardize_seq(seq: str) -> str:
    seq_u = seq.upper().replace('T', 'U')
    invalid = set(seq_u) - set('AUGC')
    if invalid:
        logging.warning(f"Sequence contains invalid characters: {invalid}")
    return seq_u


def complementary_type(seq1: str, seq2: str) -> str:
    canon = {'AU','UA','GC','CG'}
    wobble = {'GU','UG'}
    c = sum(1 for a,b in zip(seq1, seq2) if a+b in canon)
    w = sum(1 for a,b in zip(seq1, seq2) if a+b in wobble)
    if c == 6:
        return 'canonical'
    if c + w >= 5:
        return 'non-canonical'
    return 'NT'


def extract_cts(miranda_path: Path) -> dict:
    cts = {}
    lines = miranda_path.read_text().splitlines()
    info, query, ref = [], [], []
    for ln in lines:
        if ln.startswith('>'):
            info.append(ln.strip())
        if ln.strip().startswith('Query:'):
            query.append(ln.strip())
        if ln.strip().startswith('Ref:'):
            ref.append(ln.strip())
    for i, (q, r, inf) in enumerate(zip(query, ref, info)):
        seed27, seed27_site = q[-10:-4], r[-10:-4]
        seed38, seed38_site = q[-11:-5], r[-11:-5]
        t27 = complementary_type(seed27, seed27_site)
        t38 = complementary_type(seed38, seed38_site)
        if t27 == 'canonical' or t38 == 'canonical':
            btype = 'canonical'
        elif t27 == 'non-canonical' or t38 == 'non-canonical':
            btype = 'non-canonical'
        else:
            continue
        parts = inf.split()
        loc = [int(parts[6]), int(parts[7])]
        cts[str(i)] = {
            'miRNA35': q.split()[2],
            'mRNA53': r.split()[2],
            'mRNA_loc': loc,
            'binding_type': btype
        }
    return cts


def run_miranda(miRNA_seq: str, mRNA_seq: str) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        (tmp/'miRNA.fa').write_text(f">miRNA_tmp\n{miRNA_seq}\n")
        (tmp/'mRNA.fa').write_text(f">mRNA_tmp\n{mRNA_seq}\n")
        outp = tmp/'out.miranda'
        cmd = ['miranda', str(tmp/'miRNA.fa'), str(tmp/'mRNA.fa'), '-en','10000','-sc','60','-out',str(outp)]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            logging.error(f"miRanda failed for sequence pair: {e.stderr.decode().strip()}")
            return {}
        if not outp.exists():
            logging.warning("miRanda output missing for sequence pair.")
            return {}
        return extract_cts(outp)


def get_features(cts: dict, mRNA_seq: str) -> np.ndarray:
    feats = get_all_RNA_site_level_features(
        cts['miRNA35'], cts['mRNA53'], mRNA_seq, cts['mRNA_loc'], flank_number=70
    )
    return np.fromiter(feats.values(), dtype='float32')


def predict_prob(cts: dict, mRNA_seq: str, model, scaler) -> float:
    arr = get_features(cts, mRNA_seq).reshape(1, -1)
    scaled = scaler.transform(arr)
    return float(model.predict(scaled)[0, 0])


def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    fmeasure = f1_score(y_true, y_pred)
    sens = tp/(tp+fn) if (tp+fn)>0 else 0.0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0.0
    ppv = precision_score(y_true, y_pred)
    npv = tn/(tn+fn) if (tn+fn)>0 else 0.0
    return {'accuracy':acc,'Fmeasure':fmeasure,'sensitivity':sens,'specificity':spec,'PPV':ppv,'NPV':npv}


def main():
    parser = argparse.ArgumentParser(description="DeepMirTar v2 with progress and logging")
    parser.add_argument('-c','--csv',required=True,help='TSV with miRNA_id,miRNA_seq,target_id,target_seq[,label]')
    parser.add_argument('-o','--output',required=True,help='Output .tspr file')
    parser.add_argument('-t','--threshold',type=float,default=0.5,help='Score threshold')
    parser.add_argument('--metrics',help='TSV for metrics output')
    args = parser.parse_args()

    # configure logging to file and console
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('deepmirtar.log')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(ch)

    # load model and scaler
    try:
        scaler = pickle.load(open('source/MinMaxScaler.pkl','rb'), encoding='iso-8859-1')
    except Exception as e:
        logging.critical(f"Failed to load scaler: {e}")
        sys.exit(1)
    try:
        model = tf.keras.models.load_model('sda.h5')
    except Exception as e:
        logging.critical(f"Failed to load model: {e}")
        sys.exit(1)

    df = pd.read_csv(args.csv, sep='\t', header=0)
    total = len(df)
    has_label = 'label' in df.columns
    logging.info(f"Loaded {total} interactions. Has label: {has_label}")

    out_fh = open(args.output,'w')
    y_true, y_pred = [], []

    for i, row in enumerate(df.itertuples(index=False), start=1):
        mi_id, mi_seq, tid, mr_seq = row.miRNA_id, row.miRNA_seq, row.target_id, row.target_seq
        key = f"> {mi_id}|{tid}"
        true_lbl = None
        if has_label:
            lbl = getattr(row,'label',None)
            if pd.notna(lbl):
                true_lbl = int(lbl)

        mi_seq_s = standardize_seq(mi_seq)
        mr_seq_s = standardize_seq(mr_seq)
        cts = run_miranda(mi_seq_s, mr_seq_s)
        best_score, best_ct = 0.0, None
        for c in cts.values():
            try:
                p = predict_prob(c, mr_seq_s, model, scaler)
            except Exception as e:
                logging.error(f"Failed to compute features/prediction for {key}: {e}")
                continue
            if p>best_score:
                best_score, best_ct = p, c

        out_fh.write(key+"\n")
        if true_lbl is not None:
            out_fh.write(f"label: {true_lbl}\n")

        pred_lbl = 0
        if best_ct and best_score>=args.threshold:
            out_fh.write(f"Target location: {best_ct['mRNA_loc'][0]},{best_ct['mRNA_loc'][1]}\n")
            out_fh.write(f"Type: {best_ct['binding_type']}\n")
            out_fh.write(f"miRNA: {best_ct['miRNA35']} 3'-->5'\n")
            out_fh.write(f"Target: {best_ct['mRNA53']} 5'-->3'\n")
            out_fh.write(f"prediction_score: {best_score:.4f}\n\n")
            pred_lbl=1
        else:
            out_fh.write("target site:  no hits found\n\n")

        if true_lbl is not None:
            y_true.append(true_lbl)
            y_pred.append(pred_lbl)

        logging.info(f"Processed {i}/{total} pairs")

    out_fh.close()

    if has_label and y_true:
        metrics = compute_metrics(y_true, y_pred)
        logging.info("Performance metrics:")
        for k,v in metrics.items():
            logging.info(f"  {k}: {v:.4f}")
        if args.metrics:
            pd.DataFrame([metrics]).to_csv(args.metrics, sep='\t', index=False)
            logging.info(f"Metrics written to {args.metrics}")

if __name__=='__main__':
    main()
