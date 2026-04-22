# compare_all_models.py
"""
Combined model comparison: XGBoost, Random Forest, Adversarial MLP, Fair-CNN (Adversarial CNN),
TabNet, LSTM, GRU

Run: python compare_all_models.py
Requirements:
    pip install pandas scikit-learn imbalanced-learn joblib tensorflow fairlearn xgboost matplotlib seaborn pytorch-tabnet torch
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
                             classification_report, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN
import xgboost as xgb
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, losses
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import torch 

# TabNet (PyTorch implementation)
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except Exception as e:
    TabNetClassifier = None

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
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

# RNN specific
RNN_EPOCHS = 50
RNN_BATCH_SIZE = 128
RNN_DROPOUT = 0.2

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

def create_features(df):
    df = df.copy()
    df['overall_performance'] = (df['performance_score'] + df['manager_feedback'] +
                                 df['peer_feedback'] + df['kpi_score']) / 4
    df['promotion_rate'] = df['previous_promotions'] / (df['years_in_company'] + 1)
    df['years_since_last_promotion'] = df.get('promotion_year', np.nan) - df['last_promotion_year']
    df['years_since_last_promotion'] = df['years_since_last_promotion'].fillna(df['years_in_company'])
    df['training_per_year'] = df['training_hours_completed'] / (df['years_in_company'] + 1)
    df['cert_per_year'] = df['certifications_count'] / (df['years_in_company'] + 1)
    df['leadership_score'] = (df['team_collaboration_score'] + df['mentorship_score'] +
                              df['internal_initiatives']) / 3
    df['risk_score'] = df['disciplinary_actions'] - df['attendance_rate'] / 100
    df['high_performer'] = ((df['overall_performance'] > df['overall_performance'].quantile(0.75)) &
                             (df['innovation_score'] > df['innovation_score'].quantile(0.6))).astype(int)
    
    new_numeric_cols = ['overall_performance', 'promotion_rate', 'years_since_last_promotion',
                       'training_per_year', 'cert_per_year', 'leadership_score', 'risk_score',
                       'high_performer']
    
    return df, NUMERIC_COLS + new_numeric_cols

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
# Adversarial Network Components (MLP)
# -----------------------------
class GradientReversalLayer(layers.Layer):
    def __init__(self, hp_lambda=1.0, **kwargs):
        super().__init__(**kwargs)
        self.hp_lambda = tf.constant(hp_lambda, dtype=tf.float32)

    def call(self, x):
        @tf.custom_gradient
        def _reverse(x):
            def grad(dy):
                return -self.hp_lambda * dy
            return x, grad
        return _reverse(x)

def build_predictor_mlp(input_shape, hidden_units=[128, 64], dropout=0.2):
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
        metrics=[[tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='prec'), tf.keras.metrics.Recall(name='rec')],
                 [tf.keras.metrics.CategoricalAccuracy(name='accuracy')]]
    )
    
    s_map = {v: i for i, v in enumerate(classes)}
    s_train_idx = np.array([s_map[v] for v in s_train])
    s_val_idx = np.array([s_map[v] for v in s_val])
    s_train_oh = tf.keras.utils.to_categorical(s_train_idx, num_classes=n_sensitive)
    s_val_oh = tf.keras.utils.to_categorical(s_val_idx, num_classes=n_sensitive)
    
    early = EarlyStopping(monitor='val_loss', mode='min', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)
    bs = min(BATCH_SIZE, max(8, X_train.shape[0] // 5))
    
    combined_model.fit(
        X_train,
        [y_train, s_train_oh],
        validation_data=(X_val, [y_val, s_val_oh]),
        epochs=EPOCHS,
        batch_size=bs,
        callbacks=[early],
        verbose=0
    )
    
    return combined_model

# -----------------------------
# CNN-based (Fair-CNN: Adversarial CNN)
# -----------------------------
def build_predictor_cnn(input_shape, hidden_units=[128], dropout=0.2):
    inp = layers.Input(shape=input_shape, name='features_cnn')
    x = inp
    x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    
    for i, u in enumerate(hidden_units):
        x = layers.Dense(u, activation='relu', name=f'cnn_dense_{i}')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
    repr_layer = layers.Dense(64, activation='relu', name='shared_repr_cnn')(x)
    preds = layers.Dense(1, activation='sigmoid', name='promotion_output_cnn')(repr_layer)
    return Model(inputs=inp, outputs=[preds, repr_layer], name='predictor_cnn')

def build_adversary_cnn(repr_shape, n_sensitive_groups, hidden_units=[32], dropout=0.2, grl_lambda=1.0):
    repr_inp = layers.Input(shape=(repr_shape,), name='repr_input_cnn')
    x = GradientReversalLayer(hp_lambda=grl_lambda)(repr_inp)
    for i, u in enumerate(hidden_units):
        x = layers.Dense(u, activation='relu', name=f'a_dense_cnn_{i}')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
    adv_out = layers.Dense(n_sensitive_groups, activation='softmax', name='adv_output_cnn')(x)
    return Model(inputs=repr_inp, outputs=adv_out, name='adversary_cnn')

def train_adversarial_cnn(X_train, y_train, s_train, X_val, y_val, s_val, lambda_adv=LAMBDA_ADV):
    input_dim = X_train.shape[1:]
    predictor = build_predictor_cnn(input_dim)
    
    classes = np.unique(s_train)
    n_sensitive = len(classes)
    adversary = build_adversary_cnn(repr_shape=64, n_sensitive_groups=n_sensitive, grl_lambda=lambda_adv)
    
    features_in = layers.Input(shape=input_dim, name='features_in_cnn')
    preds, repr_vec = predictor(features_in)
    adv_preds = adversary(repr_vec)
    combined_model = Model(inputs=features_in, outputs=[preds, adv_preds], name='combined_cnn')
    
    combined_model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=[losses.BinaryCrossentropy(), losses.CategoricalCrossentropy()],
        loss_weights=[1.0, lambda_adv],
        metrics=[[tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='prec'), tf.keras.metrics.Recall(name='rec')],
                 [tf.keras.metrics.CategoricalAccuracy(name='accuracy')]]
    )
    
    s_map = {v: i for i, v in enumerate(classes)}
    s_train_idx = np.array([s_map[v] for v in s_train])
    s_val_idx = np.array([s_map[v] for v in s_val])
    s_train_oh = tf.keras.utils.to_categorical(s_train_idx, num_classes=n_sensitive)
    s_val_oh = tf.keras.utils.to_categorical(s_val_idx, num_classes=n_sensitive)
    
    early = EarlyStopping(monitor='val_loss', mode='min', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)
    bs = min(BATCH_SIZE, max(8, X_train.shape[0] // 5))
    
    combined_model.fit(
        X_train,
        [y_train, s_train_oh],
        validation_data=(X_val, [y_val, s_val_oh]),
        epochs=EPOCHS,
        batch_size=bs,
        callbacks=[early],
        verbose=0
    )
    
    return combined_model

# -----------------------------
# RNN models: LSTM and GRU
# -----------------------------
def build_rnn_model(cell_type='lstm', input_shape=(None,1), hidden_units=[128, 64], dropout=0.2, bidirectional=False):
    """
    cell_type: 'lstm' or 'gru'
    input_shape: (timesteps, features_per_step) -> for us (n_features, 1)
    returns: compiled Keras model (binary output)
    """
    inp = layers.Input(shape=input_shape, name=f'{cell_type}_input')
    x = inp
    # first recurrent layer returns sequences if we have more layers
    if len(hidden_units) > 1:
        if cell_type == 'lstm':
            r = layers.LSTM(hidden_units[0], return_sequences=True)
        else:
            r = layers.GRU(hidden_units[0], return_sequences=True)
        if bidirectional:
            x = layers.Bidirectional(r)(x)
        else:
            x = r(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
        # second recurrent / final
        if cell_type == 'lstm':
            r2 = layers.LSTM(hidden_units[1])
        else:
            r2 = layers.GRU(hidden_units[1])
        if bidirectional:
            x = layers.Bidirectional(r2)(x)
        else:
            x = r2(x)
    else:
        # single recurrent layer
        if cell_type == 'lstm':
            r = layers.LSTM(hidden_units[0])
        else:
            r = layers.GRU(hidden_units[0])
        if bidirectional:
            x = layers.Bidirectional(r)(x)
        else:
            x = r(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation='sigmoid', name=f'{cell_type}_output')(x)
    model = Model(inputs=inp, outputs=out, name=f'{cell_type}_model')
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name='auc')])
    return model

def train_rnn_model(X_train, y_train, X_val, y_val, cell_type='lstm', epochs=RNN_EPOCHS, batch_size=RNN_BATCH_SIZE):
    """
    X_train: 2D numpy (n_samples, n_features) or already reshaped to (n_samples, timesteps, 1)
    We'll ensure X_train is 3D: (n_samples, timesteps, 1)
    """
    # ensure numpy
    X_train_np = np.array(X_train)
    X_val_np = np.array(X_val)
    
    if X_train_np.ndim == 2:
        X_train_np = X_train_np.reshape((X_train_np.shape[0], X_train_np.shape[1], 1))
    if X_val_np.ndim == 2:
        X_val_np = X_val_np.reshape((X_val_np.shape[0], X_val_np.shape[1], 1))
    
    input_shape = X_train_np.shape[1:]
    model = build_rnn_model(cell_type=cell_type, input_shape=input_shape, hidden_units=[128, 64], dropout=RNN_DROPOUT)
    
    early = EarlyStopping(monitor='val_loss', mode='min', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)
    bs = min(batch_size, max(8, X_train_np.shape[0] // 5))
    
    model.fit(
        X_train_np, y_train,
        validation_data=(X_val_np, y_val),
        epochs=epochs,
        batch_size=bs,
        callbacks=[early],
        verbose=0
    )
    return model

# -----------------------------
# TabNet Training
# -----------------------------
def train_tabnet(X_train, y_train, X_val, y_val, max_epochs=100):
    """Train TabNet model."""
    from pytorch_tabnet.tab_model import TabNetClassifier

    clf = TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-3, weight_decay=1e-5),
        scheduler_params={"step_size":10, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        n_d=64,
        n_a=64,
        n_steps=5,
        gamma=1.5,
        lambda_sparse=1e-4,
        seed=42,
        verbose=0
    )

    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train).reshape(-1)
    X_val_np = np.array(X_val)
    y_val_np = np.array(y_val).reshape(-1)

    clf.fit(
        X_train_np, y_train_np,
        eval_set=[(X_val_np, y_val_np)],
        eval_name=["val"],
        eval_metric=["auc"],
        max_epochs=max_epochs,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )

    return clf

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
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1: {results['f1']:.4f}")
    print(f"ROC AUC: {results['roc_auc']:.4f}")
    print(f"PR AUC: {results['pr_auc']:.4f}")
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
def plot_roc_curves(all_results, y_test):
    plt.figure(figsize=(10, 8))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#3CAEA3', '#F26419', '#6A4C93', '#0B6E4F']
    for (model_name, results), color in zip(all_results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, results['y_proba'])
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={results['roc_auc']:.4f})",
                 linewidth=2.0, color=color)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, 'roc_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved ROC curves to: {save_path}")
    plt.close()

def plot_pr_curves(all_results, y_test):
    plt.figure(figsize=(10, 8))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#3CAEA3', '#F26419', '#6A4C93', '#0B6E4F']
    baseline = np.mean(y_test)
    for (model_name, results), color in zip(all_results.items(), colors):
        precisions, recalls, _ = precision_recall_curve(y_test, results['y_proba'])
        plt.plot(recalls, precisions, label=f"{model_name} (AUC={results['pr_auc']:.4f})",
                 linewidth=2.0, color=color)
    plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1.5, label=f'Random Classifier (baseline={baseline:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, 'pr_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved PR curves to: {save_path}")
    plt.close()

def plot_metrics_comparison(all_results):
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC', 'PR AUC']
    data = {model: [results[m] for m in metrics] for model, results in all_results.items()}
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(metric_labels))
    width = 0.12
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#3CAEA3', '#F26419', '#6A4C93', '#0B6E4F']
    for i, (model_name, color) in enumerate(zip(all_results.keys(), colors)):
        offset = width * (i - (len(all_results)-1)/2)
        bars = ax.bar(x + offset, data[model_name], width, label=model_name, color=color, alpha=0.9, edgecolor='black')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Metrics Comparison Across Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, 'metrics_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved metrics comparison to: {save_path}")
    plt.close()

def plot_confusion_matrices(all_results, y_test):
    n = len(all_results)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten()
    for ax, (model_name, results) in zip(axes, all_results.items()):
        cm = confusion_matrix(y_test, results['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True, square=True, linewidths=2, linecolor='black',
                   annot_kws={'size': 14, 'weight': 'bold'})
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=11, fontweight='bold')
        ax.set_xticklabels(['Not Promoted', 'Promoted'])
        ax.set_yticklabels(['Not Promoted', 'Promoted'])
    for ax in axes[len(all_results):]:
        ax.axis('off')
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, 'confusion_matrices.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrices to: {save_path}")
    plt.close()

# -----------------------------
# Main Training and Comparison
# -----------------------------
def main():
    print("="*60)
    print("MODEL COMPARISON: XGBoost vs Random Forest vs Adversarial MLP vs Fair-CNN vs TabNet vs LSTM vs GRU")
    print("="*60)
    
    # Load and prepare data
    df = load_data(DATA_PATH)
    print(f"\nLoaded data: {len(df)} rows")
    print(f"Class distribution: {df[TARGET_COL].value_counts(normalize=True).to_dict()}")
    
    df_engineered, updated_numeric_cols = create_features(df)
    df_engineered = df_engineered.drop(columns=[c for c in ID_COLS if c in df_engineered.columns], errors='ignore')
    
    X = df_engineered[updated_numeric_cols + CATEGORICAL_COLS].copy()
    y = df_engineered[TARGET_COL].astype(int).copy()
    s = df_engineered[PROTECTED_COL].copy()
    
    # Train/test split
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, s, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)
    
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Positive class ratio in train: {np.mean(y_train):.3f}")
    
    # Preprocessing
    preprocessor = build_preprocessor(updated_numeric_cols, CATEGORICAL_COLS)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    
    # Keep the test df for fairness evaluation (with original protected feature)
    df_test_for_fairness = X_test.copy()
    df_test_for_fairness['y_true'] = y_test.values
    df_test_for_fairness[PROTECTED_COL] = s_test.values
    
    # Apply SMOTEENN for class balance on training data
    print("\nApplying SMOTEENN for class balance...")
    smoteenn = SMOTEENN(random_state=RANDOM_SEED, n_jobs=-1)
    X_train_bal, y_train_bal = smoteenn.fit_resample(X_train_proc, y_train)
    print(f"After SMOTEENN: {len(X_train_bal)} samples, positive ratio: {np.mean(y_train_bal):.3f}")
    
    all_results = {}
    
    # ========== MODEL 1: XGBoost ==========
    print("\n" + "="*60)
    print("TRAINING MODEL 1: XGBoost")
    print("="*60)
    
    pos_ratio = np.sum(y_train_bal == 1) / len(y_train_bal)
    scale_pos_weight = (1 - pos_ratio) / pos_ratio if pos_ratio > 0 else 1.0
    
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        reg_alpha=1,
        reg_lambda=1,
        random_state=RANDOM_SEED,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False
    )
    
    xgb_model.fit(
        X_train_bal, y_train_bal,
        eval_set=[(X_test_proc, y_test)],
        verbose=False
    )
    
    xgb_proba = xgb_model.predict_proba(X_test_proc)[:, 1]
    xgb_threshold = find_optimal_threshold(y_test, xgb_proba)
    xgb_results = evaluate_predictions(y_test, xgb_proba, xgb_threshold)
    all_results['XGBoost'] = xgb_results
    print_model_results('XGBoost', xgb_results, y_test)
    
    # ========== MODEL 2: Random Forest ==========
    print("\n" + "="*60)
    print("TRAINING MODEL 2: Random Forest")
    print("="*60)
    
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf_model.fit(X_train_bal, y_train_bal)
    
    rf_proba = rf_model.predict_proba(X_test_proc)[:, 1]
    rf_threshold = find_optimal_threshold(y_test, rf_proba)
    rf_results = evaluate_predictions(y_test, rf_proba, rf_threshold)
    all_results['Random Forest'] = rf_results
    print_model_results('Random Forest', rf_results, y_test)
    
    # ========== MODEL 3: Adversarial Debiasing (MLP) ==========
    print("\n" + "="*60)
    print("TRAINING MODEL 3: Adversarial Debiasing (MLP)")
    print("="*60)
    
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
    
    X_adv_tr, X_adv_val, y_adv_tr, y_adv_val, s_adv_tr, s_adv_val = train_test_split(
        X_adv_bal, y_adv_bal, s_adv_bal,
        test_size=0.10, random_state=RANDOM_SEED, stratify=y_adv_bal)
    
    adv_model_mlp = train_adversarial_mlp(X_adv_tr, y_adv_tr, s_adv_tr, X_adv_val, y_adv_val, s_adv_val, lambda_adv=LAMBDA_ADV)
    
    adv_proba_test = adv_model_mlp.predict(X_test_proc, batch_size=256, verbose=0)[0].ravel()
    adv_proba_val = adv_model_mlp.predict(X_adv_val, batch_size=256, verbose=0)[0].ravel()
    adv_threshold = find_optimal_threshold(y_adv_val, adv_proba_val)
    adv_results = evaluate_predictions(y_test, adv_proba_test, adv_threshold)
    all_results['Adversarial MLP'] = adv_results
    print_model_results('Adversarial MLP', adv_results, y_test)
    
    # ========== MODEL 4: Fair-CNN (Adversarial CNN) ==========
    print("\n" + "="*60)
    print("TRAINING MODEL 4: Fair-CNN (Adversarial CNN)")
    print("="*60)
    
    X_adv_tr_cnn = X_adv_tr.reshape((X_adv_tr.shape[0], X_adv_tr.shape[1], 1))
    X_adv_val_cnn = X_adv_val.reshape((X_adv_val.shape[0], X_adv_val.shape[1], 1))
    X_test_cnn = X_test_proc.reshape((X_test_proc.shape[0], X_test_proc.shape[1], 1))
    
    adv_model_cnn = train_adversarial_cnn(X_adv_tr_cnn, y_adv_tr, s_adv_tr, X_adv_val_cnn, y_adv_val, s_adv_val, lambda_adv=LAMBDA_ADV)
    
    adv_cnn_proba_test = adv_model_cnn.predict(X_test_cnn, batch_size=256, verbose=0)[0].ravel()
    adv_cnn_proba_val = adv_model_cnn.predict(X_adv_val_cnn, batch_size=256, verbose=0)[0].ravel()
    adv_cnn_threshold = find_optimal_threshold(y_adv_val, adv_cnn_proba_val)
    adv_cnn_results = evaluate_predictions(y_test, adv_cnn_proba_test, adv_cnn_threshold)
    all_results['Fair-CNN (Adversarial)'] = adv_cnn_results
    print_model_results('Fair-CNN (Adversarial)', adv_cnn_results, y_test)
    
    # ========== MODEL 5: TabNet ==========
    print("\n" + "="*60)
    print("TRAINING MODEL 5: TabNet (pytorch-tabnet)")
    print("="*60)
    if TabNetClassifier is None:
        print("Skipping TabNet: pytorch-tabnet is not installed. Install via `pip install pytorch-tabnet torch` to enable TabNet.")
        tabnet_clf = None
    else:
        X_tab_tr, X_tab_val, y_tab_tr, y_tab_val = train_test_split(X_train_bal, y_train_bal, test_size=0.1, random_state=RANDOM_SEED, stratify=y_train_bal)
        tabnet_clf = train_tabnet(X_tab_tr, y_tab_tr, X_tab_val, y_tab_val, max_epochs=100)
        tabnet_proba_test = tabnet_clf.predict_proba(X_test_proc)[:, 1]
        tabnet_threshold = find_optimal_threshold(y_tab_val, tabnet_clf.predict_proba(X_tab_val)[:, 1])
        tabnet_results = evaluate_predictions(y_test, tabnet_proba_test, tabnet_threshold)
        all_results['TabNet'] = tabnet_results
        print_model_results('TabNet', tabnet_results, y_test)
    
    # ========== MODEL 6: LSTM ==========
    print("\n" + "="*60)
    print("TRAINING MODEL 6: LSTM (RNN treating features as sequence)")
    print("="*60)
    X_rnn_tr, X_rnn_val, y_rnn_tr, y_rnn_val = train_test_split(X_train_bal, y_train_bal, test_size=0.10, random_state=RANDOM_SEED, stratify=y_train_bal)
    lstm_clf = train_rnn_model(X_rnn_tr, y_rnn_tr, X_rnn_val, y_rnn_val, cell_type='lstm', epochs=RNN_EPOCHS, batch_size=RNN_BATCH_SIZE)
    X_test_rnn = X_test_proc.reshape((X_test_proc.shape[0], X_test_proc.shape[1], 1))
    lstm_proba_test = lstm_clf.predict(X_test_rnn, batch_size=256, verbose=0).ravel()
    lstm_proba_val = lstm_clf.predict(X_rnn_val.reshape((X_rnn_val.shape[0], X_rnn_val.shape[1], 1)), batch_size=256, verbose=0).ravel()
    lstm_threshold = find_optimal_threshold(y_rnn_val, lstm_proba_val)
    lstm_results = evaluate_predictions(y_test, lstm_proba_test, lstm_threshold)
    all_results['LSTM'] = lstm_results
    print_model_results('LSTM', lstm_results, y_test)
    
    # ========== MODEL 7: GRU ==========
    print("\n" + "="*60)
    print("TRAINING MODEL 7: GRU (RNN treating features as sequence)")
    print("="*60)
    X_rnn_tr2, X_rnn_val2, y_rnn_tr2, y_rnn_val2 = train_test_split(X_train_bal, y_train_bal, test_size=0.10, random_state=RANDOM_SEED+1, stratify=y_train_bal)
    gru_clf = train_rnn_model(X_rnn_tr2, y_rnn_tr2, X_rnn_val2, y_rnn_val2, cell_type='gru', epochs=RNN_EPOCHS, batch_size=RNN_BATCH_SIZE)
    X_test_rnn = X_test_proc.reshape((X_test_proc.shape[0], X_test_proc.shape[1], 1))
    gru_proba_test = gru_clf.predict(X_test_rnn, batch_size=256, verbose=0).ravel()
    gru_proba_val = gru_clf.predict(X_rnn_val2.reshape((X_rnn_val2.shape[0], X_rnn_val2.shape[1], 1)), batch_size=256, verbose=0).ravel()
    gru_threshold = find_optimal_threshold(y_rnn_val2, gru_proba_val)
    gru_results = evaluate_predictions(y_test, gru_proba_test, gru_threshold)
    all_results['GRU'] = gru_results
    print_model_results('GRU', gru_results, y_test)
    
    # ========== SAVE ALL MODELS ==========
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    # Save XGBoost
    xgb_model_path = os.path.join(MODELS_DIR, "xgboost_model.joblib")
    joblib.dump(xgb_model, xgb_model_path)
    print(f"Saved XGBoost model to: {xgb_model_path}")
    
    # Save Random Forest
    rf_model_path = os.path.join(MODELS_DIR, "random_forest_model.joblib")
    joblib.dump(rf_model, rf_model_path)
    print(f"Saved Random Forest model to: {rf_model_path}")
    
    # Save Adversarial MLP (Keras model - use .keras format)
    adv_mlp_model_path = os.path.join(MODELS_DIR, "adversarial_mlp_model.keras")
    adv_model_mlp.save(adv_mlp_model_path)
    print(f"Saved Adversarial MLP model to: {adv_mlp_model_path}")
    
    # Save Fair-CNN (Keras model)
    adv_cnn_model_path = os.path.join(MODELS_DIR, "fair_cnn_model.keras")
    adv_model_cnn.save(adv_cnn_model_path)
    print(f"Saved Fair-CNN model to: {adv_cnn_model_path}")
    
    # Save TabNet (if available)
    if tabnet_clf is not None:
        tabnet_model_path = os.path.join(MODELS_DIR, "tabnet_model.zip")
        tabnet_clf.save_model(tabnet_model_path)
        print(f"Saved TabNet model to: {tabnet_model_path}")
    
    # Save LSTM (Keras model)
    lstm_model_path = os.path.join(MODELS_DIR, "lstm_model.keras")
    lstm_clf.save(lstm_model_path)
    print(f"Saved LSTM model to: {lstm_model_path}")
    
    # Save GRU (Keras model)
    gru_model_path = os.path.join(MODELS_DIR, "gru_model.keras")
    gru_clf.save(gru_model_path)
    print(f"Saved GRU model to: {gru_model_path}")
    
    # Save thresholds
    thresholds = {
        'XGBoost': xgb_threshold,
        'Random Forest': rf_threshold,
        'Adversarial MLP': adv_threshold,
        'Fair-CNN (Adversarial)': adv_cnn_threshold,
        'LSTM': lstm_threshold,
        'GRU': gru_threshold
    }
    if tabnet_clf is not None:
        thresholds['TabNet'] = tabnet_threshold
    
    thresholds_path = os.path.join(MODELS_DIR, "model_thresholds.joblib")
    joblib.dump(thresholds, thresholds_path)
    print(f"Saved model thresholds to: {thresholds_path}")
    
    print(f"\nAll models saved to: {MODELS_DIR}")
    
    # ========== COMPARISON SUMMARY ==========
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    comparison_df = pd.DataFrame({
        model: {
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1': results['f1'],
            'ROC AUC': results['roc_auc'],
            'PR AUC': results['pr_auc']
        }
        for model, results in all_results.items()
    }).T
    
    print("\n" + comparison_df.to_string())
    
    best_model_name = max(all_results.keys(), key=lambda k: all_results[k]['f1'])
    print(f"\n{'='*60}")
    print(f"BEST MODEL (by F1 score): {best_model_name}")
    print(f"F1 Score: {all_results[best_model_name]['f1']:.4f}")
    print(f"{'='*60}")
    
    # ========== GENERATE VISUALIZATIONS ==========
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    plot_roc_curves(all_results, y_test)
    plot_pr_curves(all_results, y_test)
    plot_metrics_comparison(all_results)
    plot_confusion_matrices(all_results, y_test)
    
    # Fairness reporting per model
    for model_name, results in all_results.items():
        df_test_out = df_test_for_fairness.copy()
        df_test_out['y_proba'] = results['y_proba']
        df_test_out['y_pred'] = results['y_pred']
        print_group_fairness(df_test_out, model_name, protected_col=PROTECTED_COL)
    
    print("\nAll plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()