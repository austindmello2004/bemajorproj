import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

TARGET_COL = "promotion_received"
ID_COL = "employee_id"

NUMERIC_COLS = [
    "years_in_company", "current_salary", "previous_promotions",
    "last_promotion_year", "performance_score", "kpi_score",
    "manager_feedback", "peer_feedback", "training_hours_completed",
    "certifications_count", "innovation_score", "attendance_rate",
    "overtime_hours", "internal_initiatives", "team_collaboration_score",
    "mentorship_score", "disciplinary_actions", "project_delivery_success_rate",
    "client_feedback_score", "skill_progression_score", "cross_functional_exposure",
    "workload_balance", "learning_agility", "department_budget_factor",
    "vacant_positions_in_level", "salary_increase_percent"
]

CATEGORICAL_COLS = [
    "department", "education_level", "job_level", "policy_override_flag",
    "remote_work_eligibility"
]

def load_data(path):
    df = pd.read_csv(path)
    if df[TARGET_COL].dtype == "bool":
        df[TARGET_COL] = df[TARGET_COL].astype(int)
    if df[TARGET_COL].dtype == object:
        df[TARGET_COL] = df[TARGET_COL].map({"True": 1, "False": 0}).astype(int)
    return df

def build_preprocessor():
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe)
    ])
    return ColumnTransformer([
        ("num", num_pipeline, NUMERIC_COLS),
        ("cat", cat_pipeline, CATEGORICAL_COLS)
    ])
