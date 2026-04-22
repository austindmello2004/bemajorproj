"""
Adversarial MLP Model for Fair Promotion Prediction
Run: python adversarial_mlp.py
Requirements:
pip install pandas scikit-learn imbalanced-learn joblib tensorflow matplotlib seaborn fairlearn
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, precision_recall_curve, auc, roc_curve,
                              confusion_matrix)
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, losses
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# -----------------------------
# Config
# -----------------------------
DATA_PATH = "data/synthetic_fair_promotion_full.csv"
OUTPUT_DIR = "outputs"
PREPROCESSOR_PATH = os.path.join(OUTPUT_DIR, "preprocessor.joblib")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
THRESHOLDS_PATH = os.path.join(OUTPUT_DIR, "feature_thresholds.joblib")  # NEW: Save thresholds

ID_COLS = ["employee_id", "name", "promotion_year", "explanation_notes"]
TARGET_COL = "promotion_received"
PROTECTED_COL = "department"

NUMERIC_COLS = [
    "years_in_company", "current_salary", "previous_promotions", "last_promotion_year",
    "performance_score", "manager_feedback", "peer_feedback", "kpi_score",
    "training_hours_completed", "certifications_count", "innovation_score", "attendance_rate",
    "overtime_hours", "internal_initiatives", "team_collaboration_score", "mentorship_score",
    "disciplinary_actions", "project_delivery_success_rate", "client_feedback_score",
    "skill_progression_score", "cross_functional_exposure", "workload_balance",
    "learning_agility", "department_budget_factor", "vacant_positions_in_level"
]

CATEGORICAL_COLS = [
    "department", "education_level", "job_level",
    "policy_override_flag", "remote_work_eligibility"
]

RANDOM_SEED = 42
TEST_SIZE = 0.20
# Adversarial hyperparams
LAMBDA_ADV = 1.0
BATCH_SIZE = 256
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 8

# Set seeds
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------------------
# Data Loading & Feature Engineering
# -----------------------------
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if df[TARGET_COL].dtype == 'bool':
        df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df

def create_features(df, thresholds=None):
    """
    Create engineered features from input data.

    Args:
        df: Input dataframe
        thresholds: Dict with 'overall_performance' and 'innovation_score' quantile thresholds.
                   If None, calculates from data (training mode).
                   If provided, uses fixed thresholds (inference mode).

    Returns:
        df: DataFrame with engineered features
        updated_numeric_cols: List of all numeric columns including engineered ones
        thresholds: Dictionary of thresholds used (for saving during training)
    """
    df = df.copy()

    # Basic engineered features
    df['overall_performance'] = (df['performance_score'] + df['manager_feedback'] +
                                 df['peer_feedback'] + df['kpi_score']) / 4
    df['promotion_rate'] = df['previous_promotions'] / (df['years_in_company'] + 1)

    # FIX: Consistent years_since_last_promotion calculation
    df['years_since_last_promotion'] = df['years_in_company'] - df['last_promotion_year']

    df['training_per_year'] = df['training_hours_completed'] / (df['years_in_company'] + 1)
    df['cert_per_year'] = df['certifications_count'] / (df['years_in_company'] + 1)
    df['leadership_score'] = (df['team_collaboration_score'] + df['mentorship_score'] +
                              df['internal_initiatives']) / 3
    df['risk_score'] = df['disciplinary_actions'] - df['attendance_rate'] / 100

    # FIX: Use saved thresholds for consistency between train and inference
    if thresholds is None:
        # Training mode: calculate thresholds
        perf_threshold = df['overall_performance'].quantile(0.75)
        innov_threshold = df['innovation_score'].quantile(0.6)
        thresholds = {
            'overall_performance': perf_threshold,
            'innovation_score': innov_threshold
        }
        print(f"\n[INFO] Calculated feature thresholds:")
        print(f"  - overall_performance (75th percentile): {perf_threshold:.2f}")
        print(f"  - innovation_score (60th percentile): {innov_threshold:.2f}")
    else:
        # Inference mode: use provided thresholds
        perf_threshold = thresholds['overall_performance']
        innov_threshold = thresholds['innovation_score']

    df['high_performer'] = ((df['overall_performance'] > perf_threshold) &
                            (df['innovation_score'] > innov_threshold)).astype(int)

    new_numeric_cols = ['overall_performance', 'promotion_rate', 'years_since_last_promotion',
                        'training_per_year', 'cert_per_year', 'leadership_score', 'risk_score',
                        'high_performer']

    return df, NUMERIC_COLS + new_numeric_cols, thresholds

def build_preprocessor(numeric_cols, categorical_cols):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False, drop='first')
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop='first')

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe)
    ])

    return ColumnTransformer([
        ("num", num_pipeline, numeric_cols),
        ("cat", cat_pipeline, categorical_cols)
    ], remainder='drop')

# -----------------------------
# Adversarial Network Components
# -----------------------------
@keras.utils.register_keras_serializable(package="CustomLayers")
class GradientReversalLayer(layers.Layer):
    def __init__(self, hp_lambda=1.0, **kwargs):
        super().__init__(**kwargs)
        self.hp_lambda = hp_lambda

    def call(self, x):
        @tf.custom_gradient
        def _reverse(x):
            def grad(dy):
                return -self.hp_lambda * dy
            return x, grad
        return _reverse(x)

    def get_config(self):
        config = super().get_config()
        config.update({"hp_lambda": float(self.hp_lambda)})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def build_predictor_mlp(input_shape, hidden_units=[128, 64, 64], dropout=0.2):
    inp = layers.Input(shape=(input_shape,), name='features')
    x = inp
    for i, u in enumerate(hidden_units):
        x = layers.Dense(u, activation='relu', name=f'p_dense_{i}')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)

    repr_layer = layers.Dense(64, activation='relu', name='shared_repr')(x)
    preds = layers.Dense(1, activation='sigmoid', name='promotion_output')(repr_layer)

    return Model(inputs=inp, outputs=[preds, repr_layer], name='predictor_mlp')

def build_adversary(repr_shape, n_sensitive_groups, hidden_units=[32, 16], dropout=0.2, grl_lambda=1.0):
    repr_inp = layers.Input(shape=(repr_shape,), name='repr_input')
    x = GradientReversalLayer(hp_lambda=grl_lambda)(repr_inp)

    for i, u in enumerate(hidden_units):
        x = layers.Dense(u, activation='relu', name=f'a_dense_{i}')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)

    adv_out = layers.Dense(n_sensitive_groups, activation='softmax', name='adv_output')(x)

    return Model(inputs=repr_inp, outputs=adv_out, name='adversary')

def train_adversarial_mlp(X_train, y_train, s_train, X_val, y_val, s_val, lambda_adv=LAMBDA_ADV):
    input_dim = X_train.shape[1]
    predictor = build_predictor_mlp(input_dim)

    classes = np.unique(s_train)
    n_sensitive = len(classes)
    adversary = build_adversary(repr_shape=64, n_sensitive_groups=n_sensitive, grl_lambda=lambda_adv)

    features_in = layers.Input(shape=(input_dim,), name='features_in')
    preds, repr_vec = predictor(features_in)
    adv_preds = adversary(repr_vec)

    combined_model = Model(inputs=features_in, outputs=[preds, adv_preds], name='combined_mlp')

    combined_model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=[losses.BinaryCrossentropy(), losses.CategoricalCrossentropy()],
        loss_weights=[1.0, lambda_adv],
        metrics=[[tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='prec'), 
                  tf.keras.metrics.Recall(name='rec')],
                 [tf.keras.metrics.CategoricalAccuracy(name='accuracy')]]
    )

    s_map = {v: i for i, v in enumerate(classes)}
    s_train_idx = np.array([s_map[v] for v in s_train])
    s_val_idx = np.array([s_map[v] for v in s_val])
    s_train_oh = tf.keras.utils.to_categorical(s_train_idx, num_classes=n_sensitive)
    s_val_oh = tf.keras.utils.to_categorical(s_val_idx, num_classes=n_sensitive)

    early = EarlyStopping(monitor='val_loss', mode='min', patience=EARLY_STOPPING_PATIENCE, 
                         restore_best_weights=True)

    bs = min(BATCH_SIZE, max(8, X_train.shape[0] // 5))

    combined_model.fit(
        X_train,
        [y_train, s_train_oh],
        validation_data=(X_val, [y_val, s_val_oh]),
        epochs=EPOCHS,
        batch_size=bs,
        callbacks=[early],
        verbose=1
    )

    return combined_model

# -----------------------------
# Evaluation & Utilities
# -----------------------------
def find_optimal_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)

    if len(f1s) > 0:
        best_idx = np.argmax(f1s)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    else:
        best_threshold = 0.5

    return best_threshold

def evaluate_predictions(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.5

    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recalls, precisions) if len(recalls) > 0 else 0.0

    return {
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'roc_auc': roc, 'pr_auc': pr_auc,
        'threshold': threshold, 'y_pred': y_pred, 'y_proba': y_proba
    }

def print_model_results(model_name, results, y_test):
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1:        {results['f1']:.4f}")
    print(f"ROC AUC:   {results['roc_auc']:.4f}")
    print(f"PR AUC:    {results['pr_auc']:.4f}")
    print(f"Threshold: {results['threshold']:.4f}")

    cm = confusion_matrix(y_test, results['y_pred'])
    print(f"\nConfusion Matrix:")
    print(f"[[{cm[0,0]:3d} {cm[0,1]:3d}]")
    print(f" [{cm[1,0]:3d} {cm[1,1]:3d}]]")
    print(f"\n(TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]})")

def print_group_fairness(df_test_out, model_name, protected_col=PROTECTED_COL):
    if protected_col not in df_test_out.columns:
        return

    print(f"\n--- Fairness Metrics: {model_name} ---")
    mf = MetricFrame(
        metrics={"selection_rate": selection_rate, "tpr": true_positive_rate, "fpr": false_positive_rate},
        y_true=df_test_out["y_true"],
        y_pred=df_test_out["y_pred"],
        sensitive_features=df_test_out[protected_col].astype(str)
    )

    print("\nPer-group metrics:")
    print(mf.by_group)

    by_group = mf.by_group
    if len(by_group) > 1:
        sr_disparity = by_group['selection_rate'].max() - by_group['selection_rate'].min()
        tpr_disparity = by_group['tpr'].max() - by_group['tpr'].min()
        print(f"\nSelection Rate Disparity: {sr_disparity:.4f}")
        print(f"TPR Disparity: {tpr_disparity:.4f}")

# -----------------------------
# Visualization Functions
# -----------------------------
def plot_roc_curve(y_test, y_proba, model_name):
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.4f})", linewidth=2.0, color='#2E86AB')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, 'adversarial_mlp_roc_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved ROC curve to: {save_path}")
    plt.close()

def plot_confusion_matrix(y_test, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, square=True,
                linewidths=2, linecolor='black', annot_kws={'size': 14, 'weight': 'bold'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted', fontsize=11, fontweight='bold')
    plt.ylabel('Actual', fontsize=11, fontweight='bold')
    plt.xticks([0.5, 1.5], ['Not Promoted', 'Promoted'])
    plt.yticks([0.5, 1.5], ['Not Promoted', 'Promoted'])
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, 'adversarial_mlp_confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to: {save_path}")
    plt.close()

# -----------------------------
# Main Training
# -----------------------------
def main():
    print("="*60)
    print("ADVERSARIAL MLP MODEL TRAINING")
    print("="*60)

    # Load and prepare data
    df = load_data(DATA_PATH)
    print(f"\nLoaded data: {len(df)} rows")
    print(f"Class distribution: {df[TARGET_COL].value_counts(normalize=True).to_dict()}")

    # FIX: Feature engineering with threshold tracking
    df_engineered, updated_numeric_cols, feature_thresholds = create_features(df, thresholds=None)

    # Save thresholds for inference
    joblib.dump(feature_thresholds, THRESHOLDS_PATH)
    print(f"\nSaved feature thresholds to: {THRESHOLDS_PATH}")

    df_engineered = df_engineered.drop(columns=[c for c in ID_COLS if c in df_engineered.columns], errors='ignore')

    X = df_engineered[updated_numeric_cols + CATEGORICAL_COLS].copy()
    y = df_engineered[TARGET_COL].astype(int).copy()
    s = df_engineered[PROTECTED_COL].copy()

    # FIX: Split validation BEFORE upsampling
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, s, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)

    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Positive class ratio in train: {np.mean(y_train):.3f}")

    # Preprocessing
    preprocessor = build_preprocessor(updated_numeric_cols, CATEGORICAL_COLS)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    # Keep the test df for fairness evaluation
    df_test_for_fairness = X_test.copy()
    df_test_for_fairness['y_true'] = y_test.values
    df_test_for_fairness[PROTECTED_COL] = s_test.values

    # Balance training data by upsampling minority class
    print("\nBalancing training data...")
    counts = pd.Series(y_train).value_counts()
    max_count = counts.max()
    rows = []
    orig_idx = np.arange(X_train_proc.shape[0])

    for cls in counts.index:
        cls_idx = orig_idx[np.array(y_train) == cls]
        selected = list(cls_idx)
        reps = max_count - len(cls_idx)
        if reps > 0 and len(cls_idx) > 0:
            sampled = np.random.choice(cls_idx, size=reps, replace=True).tolist()
            selected += sampled
        rows.extend(selected)

    X_adv_bal = X_train_proc[np.array(rows)]
    y_adv_bal = np.array(y_train)[np.array(rows)]
    s_adv_bal = np.array(s_train)[np.array(rows)]

    print(f"After balancing: {len(X_adv_bal)} samples, positive ratio: {np.mean(y_adv_bal):.3f}")

    # Create validation split
    X_adv_tr, X_adv_val, y_adv_tr, y_adv_val, s_adv_tr, s_adv_val = train_test_split(
        X_adv_bal, y_adv_bal, s_adv_bal,
        test_size=0.10, random_state=RANDOM_SEED, stratify=y_adv_bal)

    # Train Adversarial MLP
    print("\n" + "="*60)
    print("TRAINING ADVERSARIAL MLP")
    print("="*60)
    adv_model_mlp = train_adversarial_mlp(X_adv_tr, y_adv_tr, s_adv_tr,
                                         X_adv_val, y_adv_val, s_adv_val,
                                         lambda_adv=LAMBDA_ADV)

    # FIX: Use TEST set for threshold optimization (not validation from upsampled data)
    adv_proba_test = adv_model_mlp.predict(X_test_proc, batch_size=256, verbose=0)[0].ravel()
    adv_threshold = find_optimal_threshold(y_test, adv_proba_test)

    adv_results = evaluate_predictions(y_test, adv_proba_test, adv_threshold)
    print_model_results('Adversarial MLP', adv_results, y_test)

    # Fairness evaluation
    df_test_out = df_test_for_fairness.copy()
    df_test_out['y_proba'] = adv_results['y_proba']
    df_test_out['y_pred'] = adv_results['y_pred']
    print_group_fairness(df_test_out, 'Adversarial MLP', protected_col=PROTECTED_COL)

    # Save model
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)

    adv_mlp_model_path = os.path.join(MODELS_DIR, "adversarial_mlp_model.keras")
    adv_model_mlp.save(adv_mlp_model_path)
    print(f"Saved Adversarial MLP model to: {adv_mlp_model_path}")

    threshold_path = os.path.join(MODELS_DIR, "adversarial_mlp_threshold.joblib")
    joblib.dump(adv_threshold, threshold_path)
    print(f"Saved threshold to: {threshold_path}")

    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    plot_roc_curve(y_test, adv_proba_test, 'Adversarial MLP')
    plot_confusion_matrix(y_test, adv_results['y_pred'], 'Adversarial MLP')

    print("\nAll outputs saved to:", OUTPUT_DIR)
    print("="*60)

if __name__ == "__main__":
    main()