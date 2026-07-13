"""
src/router/train_router.py
==========================
Advanced Publication-Grade Adaptive RAG Router Training Pipeline.


Fixes & Enhancements Incorporated:
1. Real Hybrid Features: Merges BGE question embeddings with actual structural patient features.
2. Z-Score Normalization: Fits a StandardScaler on training tabular features and applies it to val.
3. Expanded Search Space: Added min_child_weight, reg_alpha, and reg_lambda to Hyperparameter tuning.
4. CV Metadata Extraction: Captures and logs best, mean, and std Cross-Validation scores.
5. Absolute Reproducibility: Enforces Python, NumPy, and XGBoost seeds globally.
6. Comprehensive Metadata: Exports a structural router_metadata.json experiment log.
7. Multi-Format Artifacts: Saves confusion matrix, feature importance, and SHAP plots.
8. High-Resolution Latency: Computes Average, Median, P95, and P99 inference latency profiles.
9. Cost-Latency Alignment: Links evaluation cost metrics directly to real execution units.
10. Robust Baselines: Evaluates against Majority Class and Stratified Random classifiers.

NOTE: RouterConfig and HybridFeaturePipeline now live in
src/router/feature_pipeline.py — a stable, always-importable module —
so that feature_pipeline.pkl can be deserialized correctly by any
script (e.g. run_evaluation.py), not just this one. See that file's
module docstring for the full explanation.
"""

import json
import logging
import pickle
import random
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Headless execution mode for local terminals
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from src.router.feature_pipeline import RouterConfig, HybridFeaturePipeline

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ══════════════════════════════════════════════════════════════════════════════
# Determinism Setup
# ══════════════════════════════════════════════════════════════════════════════


def enforce_reproducibility(seed: int):
    """Enforces multi-layer environment determinism for scientific verification."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("AdvancedRouter")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if script is re-run in interactive sessions
    if not logger.handlers:
        fh = logging.FileHandler(out_dir / "production_training.log")
        ch = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# ══════════════════════════════════════════════════════════════════════════════
# Module 2: Metric Evaluation & Scientific Visualisation
# ══════════════════════════════════════════════════════════════════════════════


class AdvancedEvaluator:
    def __init__(self, config: RouterConfig, logger: logging.Logger, label_encoder: LabelEncoder):
        self.cfg = config
        self.logger = logger
        self.le = label_encoder
        self.classes = list(self.le.classes_)

    def calculate_cost_metric(self, y_true_labels, y_pred_labels) -> float:
        """Computes structural system routing cost penalizing incorrect choices."""
        total = 0.0
        for true_l, pred_l in zip(y_true_labels, y_pred_labels):
            cost = self.cfg.cost_matrix.get(pred_l, 10.0)
            if true_l != pred_l:
                cost += self.cfg.misclassification_penalty
            total += cost
        return total / len(y_true_labels)

    def evaluate_and_export(self, clf, X_train, y_train, X_val, y_val, df_val, train_metrics, latency_stats):
        self.logger.info("Running deep evaluation suite...")

        # Predictions & Softmax probabilities
        y_pred_idx = clf.predict(X_val)
        y_pred_proba = clf.predict_proba(X_val)

        y_true_labels = self.le.inverse_transform(y_val)
        y_pred_labels = self.le.inverse_transform(y_pred_idx)

        # 1. Advanced Structural Classification Performance Metrics
        metrics = {
            "Accuracy": float(accuracy_score(y_val, y_pred_idx)),
            "Balanced_Accuracy": float(balanced_accuracy_score(y_val, y_pred_idx)),
            "Macro_Precision": float(precision_score(y_val, y_pred_idx, average="macro", zero_division=0)),
            "Macro_Recall": float(recall_score(y_val, y_pred_idx, average="macro", zero_division=0)),
            "Macro_F1": float(f1_score(y_val, y_pred_idx, average="macro", zero_division=0)),
            "Weighted_F1": float(f1_score(y_val, y_pred_idx, average="weighted", zero_division=0))
        }

        # 2. Advanced Static Target Baseline Comparisons
        # Majority Class Baseline
        dummy_maj = DummyClassifier(strategy='prior')
        dummy_maj.fit(X_train, y_train)
        y_base_maj_idx = dummy_maj.predict(X_val)
        y_base_maj_labels = self.le.inverse_transform(y_base_maj_idx)

        # Stratified Random Baseline
        dummy_strat = DummyClassifier(strategy='stratified', random_state=self.cfg.seed)
        dummy_strat.fit(X_train, y_train)
        y_base_strat_idx = dummy_strat.predict(X_val)
        y_base_strat_labels = self.le.inverse_transform(y_base_strat_idx)

        baseline_metrics = {
            "Majority_Accuracy": float(accuracy_score(y_val, y_base_maj_idx)),
            "Majority_Macro_F1": float(f1_score(y_val, y_base_maj_idx, average="macro", zero_division=0)),
            "Stratified_Accuracy": float(accuracy_score(y_val, y_base_strat_idx)),
            "Stratified_Macro_F1": float(f1_score(y_val, y_base_strat_idx, average="macro", zero_division=0)),
        }

        # 3. Cost Evaluations
        router_cost = self.calculate_cost_metric(y_true_labels, y_pred_labels)
        maj_cost = self.calculate_cost_metric(y_true_labels, y_base_maj_labels)
        strat_cost = self.calculate_cost_metric(y_true_labels, y_base_strat_labels)

        # 4. Generate Confusion Matrix DataFrame and Export CSV
        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=self.classes)
        cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in self.classes], columns=[f"Pred_{c}" for c in self.classes])
        cm_df.to_csv(self.cfg.out_dir / "confusion_matrix.csv")

        # Plot Confusion Matrix Image
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.classes, yticklabels=self.classes)
        plt.ylabel('True Oracle Ground Truth')
        plt.xlabel('Predicted Router Selection')
        plt.title('Adaptive Router Confusion Matrix')
        plt.savefig(self.cfg.out_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 5. Feature Importance Calculations
        importance = clf.feature_importances_
        bge_dim_count = X_val.shape[1] - len(self.cfg.ehr_feature_cols)
        feat_names = [f"BGE_Dim_{i}" for i in range(bge_dim_count)] + self.cfg.ehr_feature_cols

        imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importance}).sort_values(by="Importance", ascending=False)
        imp_df.to_csv(self.cfg.out_dir / "feature_importance.csv", index=False)

        plt.figure(figsize=(9, 5))
        sns.barplot(data=imp_df.head(15), x="Importance", y="Feature", palette="viridis")
        plt.title("Top 15 Router Hybrid Feature Importances (XGBoost)")
        plt.xlabel("Gini Importance Score")
        plt.savefig(self.cfg.out_dir / "feature_importance.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 6. SHAP Value Generation for Publication Figures
        if HAS_SHAP:
            self.logger.info("Generating advanced SHAP interpretation summaries...")
            try:
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(X_val)
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_val, feature_names=feat_names, class_names=self.classes, show=False)
                plt.savefig(self.cfg.out_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
                plt.close()
            except Exception as e:
                self.logger.warning(f"Failed to generate SHAP summary plot: {str(e)}")
        else:
            self.logger.warning("SHAP library not found. Skipping shap_summary.png artifact generation.")

        # 7. Save Experimentation Metadata & Results JSON
        metadata = {
            "experiment_timestamp": datetime.now().isoformat(),
            "environment_settings": {"seed": self.cfg.seed, "embedding_model": self.cfg.embed_model},
            "dataset_profiles": {"training_size": len(X_val), "class_distribution": pd.Series(y_true_labels).value_counts().to_dict()},
            "cross_validation_performance": train_metrics.get("cv_results", "Skipped tuning"),
            "inference_latency_ms": latency_stats,
            "performance_metrics": metrics,
            "baseline_comparison_metrics": baseline_metrics,
            "cost_utility_analysis_units": {
                "router_average_cost": router_cost,
                "baseline_majority_cost": maj_cost,
                "baseline_stratified_cost": strat_cost
            },
            "optimized_hyperparameters": {k: str(v) for k, v in clf.get_params().items()}
        }
        with open(self.cfg.out_dir / "router_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        # 8. Comprehensive Predictions & Qualitative Error Analysis Logs
        df_val = df_val.copy()
        df_val["predicted_mode"] = y_pred_labels
        df_val["confidence"] = np.max(y_pred_proba, axis=1)
        df_val["is_correct"] = df_val["best_mode"] == df_val["predicted_mode"]

        for idx, cls_name in enumerate(self.classes):
            df_val[f"prob_{cls_name}"] = y_pred_proba[:, idx]

        df_val.to_csv(self.cfg.out_dir / "router_predictions.csv", index=False)

        # Isolate misclassifications ensuring structural context columns are exported
        errors = df_val[~df_val["is_correct"]][["question", "best_mode", "predicted_mode", "confidence"] + self.cfg.ehr_feature_cols]
        errors.to_csv(self.cfg.out_dir / "error_analysis.csv", index=False)

        # Output Core Performance to Terminal Stream
        self.logger.info(f"\n{'═'*60}\n PRODUCTION ROUTER PERFORMANCE SUMMARY\n{'═'*60}")
        self.logger.info(f" Overall Accuracy      : {metrics['Accuracy']:.4f} (Majority Base: {baseline_metrics['Majority_Accuracy']:.4f})")
        self.logger.info(f" Macro F1-Score        : {metrics['Macro_F1']:.4f} (Majority Base: {baseline_metrics['Majority_Macro_F1']:.4f})")
        self.logger.info(f" Balanced Accuracy     : {metrics['Balanced_Accuracy']:.4f}")
        self.logger.info(f"\n--- Cost Utility Comparison ---")
        self.logger.info(f" Router Average Cost   : {router_cost:.2f} compute units")
        self.logger.info(f" Baseline Majority Cost: {maj_cost:.2f} compute units")
        self.logger.info(f" Baseline Strat. Cost  : {strat_cost:.2f} compute units")
        self.logger.info(f"\n--- Latency Performance Profile ---")
        self.logger.info(f" Latency Profile (ms)  : Mean: {latency_stats['mean']:.2f} | P95: {latency_stats['p95']:.2f} | P99: {latency_stats['p99']:.2f}")
        self.logger.info(f"{'═'*60}")


# ══════════════════════════════════════════════════════════════════════════════
# Module 3: Model Optimization Engine
# ══════════════════════════════════════════════════════════════════════════════


class AdvancedModelTrainer:
    def __init__(self, config: RouterConfig, logger: logging.Logger):
        self.cfg = config
        self.logger = logger

    def optimize_and_train(self, X_train, y_train):
        t0 = time.time()
        weights = compute_sample_weight(class_weight='balanced', y=y_train)
        train_metadata = {}

        if self.cfg.tune_hyperparams:
            self.logger.info("Executing expanded RandomizedSearchCV across 5 stratified folds...")
            param_dist = {
                'max_depth': [3, 4, 5, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'n_estimators': [100, 150, 200, 300],
                'gamma': [0, 0.1, 0.2, 0.3],
                'min_child_weight': [1, 2, 4, 6],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [1.0, 2.0, 5.0]
            }

            base_xgb = xgb.XGBClassifier(objective='multi:softprob', random_state=self.cfg.seed, eval_metric='mlogloss')
            cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.cfg.seed)

            search = RandomizedSearchCV(
                base_xgb, param_distributions=param_dist, n_iter=30,
                scoring='f1_macro', cv=cv_strategy, verbose=1, random_state=self.cfg.seed, n_jobs=-1
            )
            search.fit(X_train, y_train, sample_weight=weights)

            self.logger.info(f"Optimal parameters identified: {search.best_params_}")
            clf = search.best_estimator_

            # Secure Cross Validation performance logs
            best_idx = search.best_index_
            train_metadata["cv_results"] = {
                "best_cv_macro_f1": float(search.best_score_),
                "mean_cv_macro_f1": float(search.cv_results_['mean_test_score'][best_idx]),
                "std_cv_macro_f1": float(search.cv_results_['std_test_score'][best_idx])
            }
        else:
            self.logger.info("Training production XGBoost with calibrated architecture defaults...")
            clf = xgb.XGBClassifier(
                objective='multi:softprob', max_depth=4, learning_rate=0.1,
                n_estimators=150, subsample=0.8, colsample_bytree=0.8,
                random_state=self.cfg.seed, eval_metric='mlogloss'
            )
            clf.fit(X_train, y_train, sample_weight=weights)
            train_metadata["cv_results"] = "Tuning parameter skipped. Used default calibrated configuration mapping."

        train_metadata["total_training_duration_s"] = time.time() - t0
        return clf, train_metadata

    def profiles_inference_latency(self, clf, X_val) -> dict:
        self.logger.info("Profiling high-resolution inference latency metrics...")
        latencies = []

        # Profile iteratively to mimic synchronous real-world production endpoint performance
        for i in range(len(X_val)):
            sample = X_val[i:i+1]
            t_start = time.perf_counter()
            _ = clf.predict(sample)
            latencies.append((time.perf_counter() - t_start) * 1000)  # Convert to milliseconds

        return {
            "mean": float(np.mean(latencies)),
            "median": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
            "max": float(np.max(latencies))
        }


# ══════════════════════════════════════════════════════════════════════════════
# Execution Entrypoint
# ══════════════════════════════════════════════════════════════════════════════


def main():
    cfg = RouterConfig()
    enforce_reproducibility(cfg.seed)
    logger = setup_logger(cfg.out_dir)

    logger.info("Initializing Advanced Adaptive RAG Router Training Framework")

    # Load Clean Parquet datasets
    def read_valid_dataset(path: Path) -> pd.DataFrame:
        df = pd.read_parquet(path)
        return df[~df["best_mode"].isin(["FAILED", "FAILED_GENERATION", "MISSING_MODES"])].copy()

    df_train = read_valid_dataset(cfg.train_path)
    df_val = read_valid_dataset(cfg.val_path)

    le = LabelEncoder()
    y_train = le.fit_transform(df_train["best_mode"])
    y_val = le.transform(df_val["best_mode"])

    # Feature Extraction with Standardization
    feature_pipeline = HybridFeaturePipeline(cfg, logger)
    X_train = feature_pipeline.fit_transform(df_train)
    X_val = feature_pipeline.transform(df_val)

    # Optimization Engine Execution
    trainer = AdvancedModelTrainer(cfg, logger)
    clf, train_metrics = trainer.optimize_and_train(X_train, y_train)

    # High-resolution Latency Profiling
    latency_stats = trainer.profiles_inference_latency(clf, X_val)

    # Complete Metric Suite & Artifact Export
    evaluator = AdvancedEvaluator(cfg, logger, le)
    evaluator.evaluate_and_export(clf, X_train, y_train, X_val, y_val, df_val, train_metrics, latency_stats)

    # Serialize finalized weights and components to disk
    clf.save_model(cfg.out_dir / "router_xgb_model.json")

    with open(cfg.out_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    with open(cfg.out_dir / "feature_pipeline.pkl", "wb") as f:
        pickle.dump(feature_pipeline, f)

    # Serialize bucket mapping dictionary individually for easy frontend / service parsing
    with open(cfg.out_dir / "bucket_encoder.json", "w") as f:
        json.dump(feature_pipeline.bucket_map, f, indent=4)

    logger.info("Pipeline executed successfully. All scientific artifacts saved to disk.")


if __name__ == "__main__":
    main()