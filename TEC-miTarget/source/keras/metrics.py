import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, brier_score_loss
from sklearn.metrics import average_precision_score, roc_auc_score


def accuracy(y_true, y_pred):
    """_summary_

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_

    Returns:
        float: Accuracy score
    """
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def sensitivity(y_true, y_pred):
    """
    Calculate Sensitivity (True Positive Rate) for a binary classification problem.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: Sensitivity score.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    return sensitivity

def specificity(y_true, y_pred):
    """
    Calculate Specificity (True Negative Rate) for a binary classification problem.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: Specificity score.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

def f1(y_true, y_pred):
    """
    Calculate F-score for a binary classification problem.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: F-score.
    """
    f1_s = f1_score(y_true, y_pred)
    return f1_s

def ppv(y_true, y_pred):
    """
    Calculate Positive Predictive Value (PPV) for a binary classification problem.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: PPV score.
    """
    ppv_s = precision_score(y_true, y_pred)
    return ppv_s

def npv(y_true, y_pred):
    """
    Calculate Negative Predictive Value (NPV) for a binary classification problem.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: NPV score.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    npv_s = tn / (tn + fn)
    return npv_s

def brier_score(y_true, y_pred):
    """
    Calculate Brier Score for a binary classification problem.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: Brier Score.
    """
    brier_score = brier_score_loss(y_true, y_pred)
    return brier_score

def auroc(y_true, y_pred):
    """
    Calculate Area Under the Receiver Operating Characteristic Curve (AUROC) for a binary classification problem.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: AUROC score.
    """
    auroc = roc_auc_score(y_true, y_pred)
    return auroc

def auprc(y_true, y_pred):
    """
    Calculate Area Under the Precision-Recall Curve (AUPRC) for a binary classification problem.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: AUPRC score.
    """
    auprc = average_precision_score(y_true, y_pred)
    return auprc