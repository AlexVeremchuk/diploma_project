This Python module provides a comprehensive toolkit for detecting and analyzing data leakage in machine learning pipelines. It implements statistical tests and visualization methods to identify four critical types of leakage:

Overlap Leakage: Duplicate samples between training and test sets.

Multi-test Leakage: Data appearing in multiple cross-validation folds.

Preprocessing Leakage: Contamination from global scaling/normalization.

Target Leakage: Features correlated with the target variable.

Designed for researchers and practitioners, the module quantifies leakage impact using metrics (Accuracy, Precision, Recall, F2-Score) and generates diagnostic visualizations.