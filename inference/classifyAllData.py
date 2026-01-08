#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch classification for Chinese-origin sequences.

Filename convention (examples):
  0000_T3_C_anon.nii.gz
  0002_T0_C_anon.nii.gz
  0003_T0_C#before_anon.nii.gz
  0001_T2B_S_anon.nii.gz

Rules:
  - T0 => thrombus (label 1)
  - T3 => thrombus free (label 0)
  - C => frontal view, S => lateral view
  - Extra tokens like 'before', 'anon', '#' are ignored

Pairs are built by matching sequence id (e.g., 0000) and T token.
Outputs match classifyAll: per-sample CSV and summary metrics (accuracy, CM, AUC, MCC).
"""

import argparse
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Classificator import Classificator

DEFAULT_DATA_DIR = \
    r"D:\ThromboMap\2025-08-28-AmTICIS-Extractor\extracted_all\FirstChannel-CorrectRange-uint16"

FNAME_RE = re.compile(
    r"^RAW_(?P<seq>\d+)[_#].*?(?P<t>T\d+[A-Z]?).*?[_#](?P<view>[CS])[^/]*?\.(nii|nii\.gz)$",
    re.IGNORECASE,
)


def parse_file_info(filename: str) -> Optional[Tuple[str, str, str]]:
    m = FNAME_RE.match(filename)
    if not m:
        return None
    seq = m.group('seq')
    t = m.group('t').upper()
    view = m.group('view').upper()
    return seq, t, view


def build_pairs(data_dir: str) -> List[dict]:
    files = [f for f in os.listdir(data_dir) if f.lower().endswith((".nii", ".nii.gz"))]
    buckets: Dict[Tuple[str, str], Dict[str, str]] = {}
    for f in files:
        info = parse_file_info(f)
        if not info:
            continue
        seq, t, view = info
        key = (seq, t)
        if key not in buckets:
            buckets[key] = {}
        buckets[key][view] = f

    entries: List[dict] = []
    for (seq, t), views in buckets.items():
        if 'C' in views and 'S' in views:
            filename = views['C']
            filename2 = views['S']
            label = 1 if t == 'T0' else 0 if t == 'T3' else None
            if label is None:
                # Unknown T-token (e.g., T2B). Skip from metrics but still can classify if desired.
                continue
            entries.append({
                'seq': seq,
                't_token': t,
                'filename': filename,
                'filenameOtherView': filename2,
                'true_label': label,
                'frontalAndLateralView': True,
            })
    return entries


def mean_activation(outputs_f: List[float], outputs_l: List[float]) -> float:
    vals: List[float] = []
    if outputs_f:
        vals.extend(outputs_f)
    if outputs_l:
        vals.extend(outputs_l)
    if len(vals) == 0:
        return 0.5
    return float(np.mean(vals))


def try_import_tqdm():
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Classify Chinese dataset using filename conventions.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="Directory containing Chinese .nii/.nii.gz files (default: %(default)s)")
    parser.add_argument("--models-dir", default="models",
                        help="Models root directory containing 'frontal' and 'lateral' subfolders.")
    parser.add_argument("--out-predictions-csv", default="autoclassify_results_chinese_data_correct_range.csv",
                        help="Output CSV path for per-sample predictions (default: %(default)s)")
    parser.add_argument("--out-metrics-csv", default="autoclassify_metrics_chinese_data_correct_range.csv",
                        help="Output CSV path for summary metrics (default: %(default)s)")

    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"Data directory not found: {data_dir}")

    entries_all = build_pairs(data_dir)
    if len(entries_all) == 0:
        raise RuntimeError("No paired C/S entries with known T tokens found.")

    clf = Classificator()
    clf.load_models(args.models_dir)
    mf_key = next(iter(clf.models_frontal.keys())) if getattr(clf, 'models_frontal', None) else ""
    ml_key = next(iter(clf.models_lateral.keys())) if getattr(clf, 'models_lateral', None) else ""

    tqdm_cls = try_import_tqdm()
    progress_iter = tqdm_cls(entries_all, desc="Classifying", unit="seq") if tqdm_cls else entries_all

    y_true: List[int] = []
    y_prob: List[float] = []
    y_pred: List[int] = []
    rows: List[dict] = []
    skipped = 0

    for entry in progress_iter:
        f_file = os.path.join(data_dir, entry['filename'])
        l_file = os.path.join(data_dir, entry['filenameOtherView'])
        if not os.path.exists(f_file) or not os.path.exists(l_file):
            skipped += 1
            continue

        try:
            clf.prepare_images(f_file, l_file, normalized=True)
            start_time = time.time()
            outputs_f, outputs_l, _, _ = clf.do_classification(f_file, l_file, mf=mf_key, ml=ml_key)
            end_time = time.time()
            classification_time = end_time - start_time
        except ValueError as e:
            msg = str(e)
            if "Height and Width of image" in msg or "is_check_shapes" in msg:
                if tqdm_cls:
                    progress_iter.write(f"Skipping (shape mismatch): {os.path.basename(f_file)} | {os.path.basename(l_file)}")
                skipped += 1
                continue
            if tqdm_cls:
                progress_iter.write(f"Skipping (ValueError): {os.path.basename(f_file)} | {os.path.basename(l_file)} | {msg}")
            skipped += 1
            continue
        except Exception as e:
            if tqdm_cls:
                progress_iter.write(f"Skipping (error): {os.path.basename(f_file)} | {os.path.basename(l_file)} | {e}")
            skipped += 1
            continue

        prob = mean_activation(outputs_f, outputs_l)
        pred = 1 if prob > 0.57 else 0
        true_label = int(entry['true_label'])

        y_true.append(true_label)
        y_prob.append(prob)
        y_pred.append(pred)

        rows.append({
            'seq': entry['seq'],
            't_token': entry['t_token'],
            'file_frontal': f_file,
            'file_lateral': l_file,
            'prob_thrombus': prob,
            'pred_label': pred,
            'true_label': true_label,
            'classification_time': classification_time
        })

    if len(rows) == 0:
        raise RuntimeError("No valid samples classified; check filenames and directory.")

    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float('nan')
    mcc = float(matthews_corrcoef(y_true, y_pred))

    df = pd.DataFrame(rows)
    df.to_csv(args.out_predictions_csv, index=False)

    metrics_df = pd.DataFrame([
        {
            'accuracy': acc,
            'auc': auc,
            'mcc': mcc,
            'tn': int(cm[0, 0]) if cm.shape == (2, 2) else None,
            'fp': int(cm[0, 1]) if cm.shape == (2, 2) else None,
            'fn': int(cm[1, 0]) if cm.shape == (2, 2) else None,
            'tp': int(cm[1, 1]) if cm.shape == (2, 2) else None,
            'num_samples': len(rows)
        }
    ])
    metrics_df.to_csv(args.out_metrics_csv, index=False)

    print(f"Accuracy: {acc:.4f}")
    if cm.shape == (2, 2):
        print(f"Confusion Matrix (tn, fp; fn, tp):\n{cm}")
    else:
        print("Confusion Matrix not available (non-binary labels encountered)")
    print(f"AUC: {auc if not np.isnan(auc) else 'nan'}")
    print(f"MCC: {mcc:.4f}")
    print(f"Skipped samples: {skipped}")
    print(f"Saved predictions to: {args.out_predictions_csv}")
    print(f"Saved metrics to: {args.out_metrics_csv}")


if __name__ == "__main__":
    main()

