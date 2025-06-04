import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
# =============================================
# 2. Leakage Detector Class with Metrics
# =============================================
class DataLeakageDetector:
    def __init__(self):
        self.results = {}
        self.ground_truth = None
        self.predictions = {
            "overlap": None,
            "multitest": None,
            "preprocessing": None,
            "target": None
        }
    
    def set_ground_truth(self, leak_flags):
        self.ground_truth = leak_flags
    
    def detect_overlap(self, X_train, X_test):
        p_values = [ks_2samp(X_train[:, i], X_test[:, i]).pvalue for i in range(X_train.shape[1])]
        leak_detected = any(p > 0.05 for p in p_values)
        self.predictions["overlap"] = leak_detected
        self.results["Overlap Leakage"] = {
            "p_values": p_values, 
            "leak_detected": leak_detected
        }
        return leak_detected
    
    def detect_multitest(self, X):
        sample_counts = np.zeros(len(X))
        kf = KFold(n_splits=5)
        for _, test_idx in kf.split(X):
            sample_counts[test_idx] += 1
        leak_detected = np.any(sample_counts > 1)
        self.predictions["multitest"] = leak_detected
        self.results["Multi-test Leakage"] = {
            "leak_detected": leak_detected
        }
        return leak_detected
    
    def detect_preprocessing(self, X_train, X_test):
        train_mean, train_std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
        test_mean, test_std = np.mean(X_test, axis=0), np.std(X_test, axis=0)
        mean_diff = np.abs(train_mean - test_mean)
        std_diff = np.abs(train_std - test_std)
        leak_detected = np.any(mean_diff < 0.1) or np.any(std_diff < 0.1)
        self.predictions["preprocessing"] = leak_detected
        self.results["Preprocessing Leakage"] = {
            "mean_diff": mean_diff, 
            "std_diff": std_diff, 
            "leak_detected": leak_detected
        }
        return leak_detected
    
    def detect_target(self, X, y):
        corr = np.array([np.abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])])
        leak_detected = np.any(corr > 0.8)
        self.predictions["target"] = leak_detected
        self.results["Target Leakage"] = {
            "correlations": corr, 
            "leak_detected": leak_detected
        }
        return leak_detected
    
    def calculate_metrics(self):
        if not self.ground_truth:
            raise ValueError("Ground truth not set. Call `set_ground_truth()` first.")
        
        # Initialize confusion matrix components
        TP = TN = FP = FN = 0
        
        # Compare predictions vs. ground truth for each leak type
        for leak_type in self.ground_truth:
            actual = self.ground_truth[leak_type]
            predicted = self.predictions[leak_type]
            
            if actual and predicted:
                TP += 1
            elif not actual and not predicted:
                TN += 1
            elif not actual and predicted:
                FP += 1
            elif actual and not predicted:
                FN += 1
        
        # Calculate metrics
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f2_score = (1 + 2**2) * (precision * recall) / (2**2 * precision + recall) if (2**2 * precision + recall) > 0 else 0
        
        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F2_Score": f2_score,
            "Confusion_Matrix": {
                "TP": TP,
                "TN": TN,
                "FP": FP,
                "FN": FN
            }
        }
        
        self.results["Metrics"] = metrics
        return metrics
    
    def visualize(self):
        """Plot leakage detection results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
       
        # Overlap Leakage
        sns.barplot(x=np.arange(len(self.results["Overlap Leakage"]["p_values"])), 
                    y=self.results["Overlap Leakage"]["p_values"], ax=axes[0, 0])
        axes[0, 0].axhline(0.05, color="red", linestyle="--")
        axes[0, 0].set_title("Overlap Leakage (KS-test p-values)")
        
        # Target Leakage
        sns.barplot(x=np.arange(len(self.results["Target Leakage"]["correlations"])), 
                    y=self.results["Target Leakage"]["correlations"], ax=axes[0, 1])
        axes[0, 1].axhline(0.8, color="red", linestyle="--")
        axes[0, 1].set_title("Target Leakage (Feature-Target Correlation)")
        
        # Preprocessing Leakage
        sns.heatmap(
            pd.DataFrame({
                "Mean Diff": self.results["Preprocessing Leakage"]["mean_diff"],
                "Std Diff": self.results["Preprocessing Leakage"]["std_diff"]
            }), 
            annot=True, ax=axes[1, 0]
        )
        axes[1, 0].set_title("Preprocessing Leakage (Mean/Std Differences)")
        
        # Adversarial AUC
        if "Adversarial AUC" in self.results:
            axes[1, 1].bar(["AUC"], [self.results["Adversarial AUC"]["AUC"]], color="skyblue")
            axes[1, 1].axhline(0.7, color="red", linestyle="--")
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].set_title("Adversarial Validation AUC")
        
        plt.tight_layout()
        plt.show()


def load_data_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    X_train = np.array(data["X_train"])
    X_test = np.array(data["X_test"])
    y_train = np.array(data["y_train"])
    y_test = np.array(data["y_test"])
    X_leaky = np.array(data["X_leaky"])
    leak_flags = data["leak_flags"]
    return X_train, X_test, y_train, y_test, X_leaky, leak_flags

