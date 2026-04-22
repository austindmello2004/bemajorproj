"""
Streamlit Frontend for Adversarial MLP Promotion Prediction
Run: streamlit run frontend.py
Requirements: pip install streamlit pandas numpy joblib tensorflow plotly openpyxl
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# -----------------------------
# Configuration
# -----------------------------
MODELS_DIR = "outputs/models"
PREPROCESSOR_PATH = "outputs/preprocessor.joblib"
MODEL_PATH = os.path.join(MODELS_DIR, "adversarial_mlp_model.keras")
THRESHOLD_PATH = os.path.join(MODELS_DIR, "adversarial_mlp_threshold.joblib")

# Column definitions
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
    "department", "education_level", "job_level", "policy_override_flag", "remote_work_eligibility"
]

# Categorical options (model expects these values)
DEPARTMENT_OPTIONS = ['Finance', 'HR', 'IT', 'Legal', 'Operations', 'Sales']
EDUCATION_OPTIONS = ['Bachelor', 'Diploma', 'Master', 'PhD']
JOB_LEVEL_OPTIONS =['Entry', 'Leadership', 'Manager', 'Mid', 'Senior']
POLICY_OPTIONS = [False, True]
REMOTE_OPTIONS = [False, True]

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Promotion Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom Layer Definition (must match training)
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

# -----------------------------
# Helper Functions
# -----------------------------
@st.cache_resource
def load_model_and_preprocessor():
    """Load the trained model, preprocessor, and threshold."""
    try:
        model = keras.models.load_model(
            MODEL_PATH,
            custom_objects={'GradientReversalLayer': GradientReversalLayer}
        )
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        threshold = joblib.load(THRESHOLD_PATH)
        return model, preprocessor, threshold
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def to_excel(df):
    """Convert DataFrame to Excel file in memory."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Predictions')
    output.seek(0)
    return output

def standardize_csv_values(df):
    """
    Standardize CSV values to match the model's expected format.
    This handles various data inconsistencies from different data sources.
    """
    df = df.copy()

    # Education Level mapping
    education_mapping = {
        'Bachelor': "Bachelor's",
        'bachelor': "Bachelor's",
        'Bachelors': "Bachelor's",
        'Master': "Master's",
        'master': "Master's",
        'Masters': "Master's",
        'Diploma': "High School",
        'diploma': "High School",
        'HighSchool': "High School",
        'high school': "High School",
        'PhD': 'PhD',
        'phd': 'PhD',
        'Doctorate': 'PhD'
    }
    df['education_level'] = df['education_level'].replace(education_mapping)

    # Job Level mapping
    job_level_mapping = {
        'Entry': 'Junior',
        'entry': 'Junior',
        'Leadership': 'Lead',
        'leadership': 'Lead',
        'lead': 'Lead',
        'junior': 'Junior',
        'mid': 'Mid',
        'senior': 'Senior',
        'manager': 'Manager',
        'Manager': 'Manager',
        'director': 'Director',
        'Director': 'Director'
    }
    df['job_level'] = df['job_level'].replace(job_level_mapping)

    # Policy Override Flag - handle boolean/string values
    if df['policy_override_flag'].dtype == bool or df['policy_override_flag'].dtype == 'bool':
        df['policy_override_flag'] = df['policy_override_flag'].map({True: 'Yes', False: 'No'})
    else:
        policy_mapping = {
            'True': 'Yes', 'true': 'Yes', 'TRUE': 'Yes', True: 'Yes', 1: 'Yes',
            'False': 'No', 'false': 'No', 'FALSE': 'No', False: 'No', 0: 'No'
        }
        df['policy_override_flag'] = df['policy_override_flag'].replace(policy_mapping)

    # Remote Work Eligibility - handle boolean/string values
    if df['remote_work_eligibility'].dtype == bool or df['remote_work_eligibility'].dtype == 'bool':
        df['remote_work_eligibility'] = df['remote_work_eligibility'].map({True: 'Eligible', False: 'Not Eligible'})
    else:
        remote_mapping = {
            'True': 'Eligible', 'true': 'Eligible', 'TRUE': 'Eligible', True: 'Eligible', 1: 'Eligible',
            'False': 'Not Eligible', 'false': 'Not Eligible', 'FALSE': 'Not Eligible', False: 'Not Eligible', 0: 'Not Eligible'
        }
        df['remote_work_eligibility'] = df['remote_work_eligibility'].replace(remote_mapping)

    return df

def create_features(df):
    """Create engineered features from input data."""
    df = df.copy()
    df['overall_performance'] = (df['performance_score'] + df['manager_feedback'] + 
                                 df['peer_feedback'] + df['kpi_score']) / 4
    df['promotion_rate'] = df['previous_promotions'] / (df['years_in_company'] + 1)
    df['years_since_last_promotion'] = df['years_in_company'] - df['last_promotion_year']
    df['training_per_year'] = df['training_hours_completed'] / (df['years_in_company'] + 1)
    df['cert_per_year'] = df['certifications_count'] / (df['years_in_company'] + 1)
    df['leadership_score'] = (df['team_collaboration_score'] + df['mentorship_score'] + 
                              df['internal_initiatives']) / 3
    df['risk_score'] = df['disciplinary_actions'] - df['attendance_rate'] / 100
    df['high_performer'] = ((df['overall_performance'] > 7.5) & 
                            (df['innovation_score'] > 7.0)).astype(int)
    return df

def make_batch_predictions(df, model, preprocessor, threshold):
    """Make predictions for batch of employees."""
    # Standardize categorical values to match model expectations
    df = standardize_csv_values(df)

    # Create features
    df_features = create_features(df)

    # Get all required columns
    all_numeric = NUMERIC_COLS + ['overall_performance', 'promotion_rate', 
                                   'years_since_last_promotion', 'training_per_year',
                                   'cert_per_year', 'leadership_score', 'risk_score', 
                                   'high_performer']

    # Prepare data
    X = df_features[all_numeric + CATEGORICAL_COLS]

    # Preprocess
    X_processed = preprocessor.transform(X)

    # Predict
    predictions = model.predict(X_processed, verbose=0)
    promotion_probas = predictions[0][:, 0]

    # Store original probabilities before applying vacant position rule
    original_probas = promotion_probas.copy()
    
    # Get vacant positions
    vacant_positions = df['vacant_positions_in_level'].values

    # Apply threshold to original probabilities
    original_preds = (original_probas >= threshold).astype(int)

    # Create three categories:
    # 0: Not Recommended
    # 1: Promoted
    # 2: Can be promoted but no vacant position
    promotion_preds = np.where(
        (original_preds == 1) & (vacant_positions == 0), 
        2,  # Eligible but no vacancy
        original_preds  # Either 0 (not recommended) or 1 (promoted)
    )

    # Set probability to 0 for those without vacant positions
    promotion_probas = np.where(vacant_positions == 0, 0.0, promotion_probas)

    return promotion_preds, promotion_probas, original_probas

# -----------------------------
# Tab 1: Individual Prediction
# -----------------------------
def individual_prediction_tab():
    st.header("🎯 Individual Employee Prediction")

    # Load model
    model, preprocessor, threshold = load_model_and_preprocessor()
    if model is None:
        st.error("⚠️ Failed to load model. Please ensure the model files exist.")
        return

    # Create two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("👤 Personal & Professional Details")
        department = st.selectbox("Department", DEPARTMENT_OPTIONS)
        education_level = st.selectbox("Education Level", EDUCATION_OPTIONS)
        job_level = st.selectbox("Job Level", JOB_LEVEL_OPTIONS)
        years_in_company = st.slider("Years in Company", 0, 30, 5)
        current_salary = st.number_input("Current Salary ($)", 30000, 10000000, 60000, step=5000)
        previous_promotions = st.number_input("Previous Promotions", 0, 10, 1)
        last_promotion_year = st.slider("Years Since Last Promotion", 0, int(years_in_company), 2)
        policy_override = st.selectbox("Policy Override Flag", POLICY_OPTIONS)
        remote_work = st.selectbox("Remote Work Eligibility", REMOTE_OPTIONS)

    with col2:
        st.subheader("📊 Performance Metrics")
        performance_score = st.slider("Performance Score (0-10)", 0.0, 10.0, 7.5, 0.1)
        manager_feedback = st.slider("Manager Feedback (0-10)", 0.0, 10.0, 7.5, 0.1)
        peer_feedback = st.slider("Peer Feedback (0-10)", 0.0, 10.0, 7.5, 0.1)
        kpi_score = st.slider("KPI Score (0-10)", 0.0, 10.0, 7.5, 0.1)
        innovation_score = st.slider("Innovation Score (0-10)", 0.0, 10.0, 7.0, 0.1)
        team_collaboration = st.slider("Team Collaboration Score (0-10)", 0.0, 10.0, 7.5, 0.1)
        mentorship_score = st.slider("Mentorship Score (0-10)", 0.0, 10.0, 7.0, 0.1)
        attendance_rate = st.slider("Attendance Rate (%)", 0.0, 100.0, 95.0, 0.5)

    # Additional metrics
    with st.expander("📈 Additional Metrics (Optional)"):
        col3, col4 = st.columns(2)
        with col3:
            training_hours = st.number_input("Training Hours Completed", 0, 500, 0)
            certifications = st.number_input("Certifications Count", 0, 20, 2)
            overtime_hours = st.number_input("Overtime Hours", 0, 1000, 0)
            internal_initiatives = st.slider("Internal Initiatives (0-10)", 0.0, 10.0, 5.0, 0.1)
            disciplinary_actions = st.number_input("Disciplinary Actions", 0, 10, 0)
        with col4:
            project_delivery = st.slider("Project Delivery Success Rate (%)", 0.0, 100.0, 85.0, 0.5)
            client_feedback = st.slider("Client Feedback Score (0-10)", 0.0, 10.0, 7.5, 0.1)
            skill_progression = st.slider("Skill Progression Score (0-10)", 0.0, 10.0, 7.0, 0.1)
            cross_functional = st.slider("Cross-functional Exposure (0-10)", 0.0, 10.0, 5.0, 0.1)
            workload_balance = st.slider("Workload Balance (0-10)", 0.0, 10.0, 7.0, 0.1)
            learning_agility = st.slider("Learning Agility (0-10)", 0.0, 10.0, 7.0, 0.1)
            dept_budget = st.slider("Department Budget Factor", 0.5, 2.0, 1.0, 0.1)
            vacant_positions = st.number_input("Vacant Positions in Level", 0, 10, 2)

    # Add warning if vacant positions is 0
    if vacant_positions == 0:
        st.warning("⚠️ No vacant positions available in this level. Promotion probability will be set to 0%.")

    # Prepare input data
    input_data = {
        'department': department, 'education_level': education_level, 'job_level': job_level,
        'years_in_company': years_in_company, 'current_salary': current_salary,
        'previous_promotions': previous_promotions, 'last_promotion_year': last_promotion_year,
        'performance_score': performance_score, 'manager_feedback': manager_feedback,
        'peer_feedback': peer_feedback, 'kpi_score': kpi_score,
        'training_hours_completed': training_hours, 'certifications_count': certifications,
        'innovation_score': innovation_score, 'attendance_rate': attendance_rate,
        'overtime_hours': overtime_hours, 'internal_initiatives': internal_initiatives,
        'team_collaboration_score': team_collaboration, 'mentorship_score': mentorship_score,
        'disciplinary_actions': disciplinary_actions,
        'project_delivery_success_rate': project_delivery,
        'client_feedback_score': client_feedback, 'skill_progression_score': skill_progression,
        'cross_functional_exposure': cross_functional, 'workload_balance': workload_balance,
        'learning_agility': learning_agility, 'department_budget_factor': dept_budget,
        'vacant_positions_in_level': vacant_positions,
        'policy_override_flag': policy_override, 'remote_work_eligibility': remote_work
    }

    # Predict button
    if st.button("🔮 Predict Promotion Probability", type="primary", width="stretch"):
        with st.spinner("Analyzing employee data..."):
            try:
                df = pd.DataFrame([input_data])
                predictions, probabilities, original_probabilities = make_batch_predictions(df, model, preprocessor, threshold)

                st.markdown("---")
                st.subheader("📋 Prediction Results")

                # Show result based on category
                if predictions[0] == 1:
                    st.success(f"✅ **Promotion Recommended** (Probability: {probabilities[0]*100:.2f}%)")
                elif predictions[0] == 2:
                    st.info(f"🔄 **Can Be Promoted - No Vacant Position** (Original Probability: {original_probabilities[0]*100:.2f}%)")
                    st.warning("⚠️ Employee qualifies for promotion but there are no vacant positions in this level.")
                else:
                    st.warning(f"⚠️ **Not Recommended for Promotion** (Probability: {probabilities[0]*100:.2f}%)")

                # Gauge chart - show original probability if category 2
                display_prob = original_probabilities[0] if predictions[0] == 2 else probabilities[0]
                gauge_title = "Promotion Probability (%)" if predictions[0] != 2 else "Eligibility Probability (No Vacancy)"
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=display_prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': gauge_title, 'font': {'size': 24}},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "orange" if predictions[0] == 2 else "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': '#ffcccc'},
                            {'range': [30, 70], 'color': '#ffffcc'},
                            {'range': [70, 100], 'color': '#ccffcc'}
                        ],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'value': threshold*100}
                    }
                ))
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

# -----------------------------
# Tab 2: Batch Prediction
# -----------------------------
def batch_prediction_tab():
    st.header("📁 Batch Employee Predictions")
    st.markdown("Upload a CSV file containing employee records to predict promotions for multiple employees.")

    # Load model
    model, preprocessor, threshold = load_model_and_preprocessor()
    if model is None:
        st.error("⚠️ Failed to load model. Please ensure the model files exist.")
        return

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

    # Show required columns
    with st.expander("📋 Required CSV Columns & Accepted Values"):
        st.markdown("Your CSV file must contain the following columns:")
        all_cols = NUMERIC_COLS + CATEGORICAL_COLS
        st.code(", ".join(all_cols))

        st.markdown("**Accepted Categorical Values:**")
        st.markdown(f"- **Education Level**: Bachelor/Bachelor's, Master/Master's, PhD, Diploma/High School")
        st.markdown(f"- **Job Level**: Entry/Junior, Mid, Senior, Leadership/Lead, Manager, Director")
        st.markdown(f"- **Policy Override**: Yes/No or True/False")
        st.markdown(f"- **Remote Work**: Eligible/Not Eligible or True/False")

    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)

            st.success(f"✅ File uploaded successfully! Found {len(df)} employee records.")

            # Show preview
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(10), width="stretch")

            # Validate columns
            missing_cols = set(NUMERIC_COLS + CATEGORICAL_COLS) - set(df.columns)
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                return

            # Check for employees with 0 vacant positions
            zero_vacant_count = (df['vacant_positions_in_level'] == 0).sum()
            if zero_vacant_count > 0:
                st.info(f"ℹ️ Found {zero_vacant_count} employee(s) with 0 vacant positions. They will be marked as 'Eligible but No Vacancy' if they qualify.")

            # Predict button
            if st.button("🚀 Run Batch Predictions", type="primary", width="stretch"):
                with st.spinner("Processing predictions for all employees..."):
                    try:
                        # Make predictions
                        predictions, probabilities, original_probabilities = make_batch_predictions(df, model, preprocessor, threshold)

                        # Add results to dataframe
                        df['promotion_probability'] = probabilities
                        df['original_probability'] = original_probabilities
                        df['promotion_recommendation'] = predictions
                        df['recommendation_label'] = df['promotion_recommendation'].map({
                            0: 'Not Recommended',
                            1: 'Promote',
                            2: 'Eligible - No Vacancy'
                        })

                        st.markdown("---")
                        st.subheader("📊 Prediction Results")

                        # Summary metrics with 3 categories
                        col1, col2, col3, col4, col5 = st.columns(5)

                        total_employees = len(df)
                        promoted = (predictions == 1).sum()
                        eligible_no_vacancy = (predictions == 2).sum()
                        not_recommended = (predictions == 0).sum()
                        avg_probability = probabilities.mean()

                        col1.metric("Total Employees", total_employees)
                        col2.metric("Promote", promoted, 
                                   delta=f"{(promoted/total_employees)*100:.1f}%")
                        col3.metric("Eligible - No Vacancy", eligible_no_vacancy,
                                   delta=f"{(eligible_no_vacancy/total_employees)*100:.1f}%")
                        col4.metric("Not Recommended", not_recommended,
                                   delta=f"{(not_recommended/total_employees)*100:.1f}%")
                        col5.metric("Avg Probability", f"{avg_probability*100:.1f}%")

                        # Check if ground truth exists
                        has_ground_truth = 'promotion_received' in df.columns
                        if has_ground_truth:
                            st.info(f"📊 Ground truth detected! Comparing predictions with actual promotions...")

                            # Calculate accuracy metrics
                            actual_promotions = df['promotion_received'].values
                            correct_predictions = (predictions == actual_promotions).sum()
                            accuracy = (correct_predictions / total_employees) * 100

                            st.markdown("### 🎯 Model Performance")
                            perf_col1, perf_col2, perf_col3 = st.columns(3)
                            perf_col1.metric("Accuracy", f"{accuracy:.1f}%")
                            perf_col2.metric("Correct Predictions", correct_predictions)
                            perf_col3.metric("Incorrect Predictions", total_employees - correct_predictions)

                        # Visualization
                        st.markdown("---")

                        # Charts in columns
                        viz_col1, viz_col2 = st.columns(2)

                        with viz_col1:
                            # Pie chart with 3 categories
                            fig_pie = px.pie(
                                values=[promoted, eligible_no_vacancy, not_recommended],
                                names=['Promote', 'Eligible - No Vacancy', 'Not Recommended'],
                                title='Promotion Recommendations Distribution',
                                color_discrete_sequence=['#2ecc71', '#f39c12', '#e74c3c']
                            )
                            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

                        with viz_col2:
                            # Histogram of probabilities
                            fig_hist = px.histogram(
                                df, x='promotion_probability',
                                nbins=20,
                                title='Distribution of Promotion Probabilities',
                                labels={'promotion_probability': 'Promotion Probability'}
                            )
                            fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red",
                                             annotation_text="Threshold")
                            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

                        # Breakdown by department
                        if 'department' in df.columns:
                            st.markdown("---")
                            st.subheader("📈 Department-wise Analysis")

                            dept_summary = df.groupby('department').agg({
                                'promotion_recommendation': ['sum', 'count'],
                                'promotion_probability': 'mean'
                            }).round(3)

                            dept_summary.columns = ['Recommended', 'Total', 'Avg Probability']
                            dept_summary['Recommendation Rate (%)'] = (
                                dept_summary['Recommended'] / dept_summary['Total'] * 100
                            ).round(1)

                            st.dataframe(dept_summary, width="stretch")

                            # Bar chart
                            fig_dept = px.bar(
                                dept_summary.reset_index(),
                                x='department',
                                y='Recommendation Rate (%)',
                                title='Promotion Recommendation Rate by Department',
                                color='Recommendation Rate (%)',
                                color_continuous_scale='Viridis'
                            )
                            st.plotly_chart(fig_dept, use_container_width=True, config={"displayModeBar": False})

                        # Show detailed results
                        st.markdown("---")
                        st.subheader("📋 Detailed Results")

                        # Filter options
                        filter_col1, filter_col2 = st.columns(2)
                        with filter_col1:
                            show_filter = st.selectbox(
                                "Filter Results",
                                ['All', 'Promote Only', 'Eligible - No Vacancy Only', 'Not Recommended Only']
                            )

                        with filter_col2:
                            sort_by = st.selectbox(
                                "Sort By",
                                ['Probability (High to Low)', 'Probability (Low to High)', 'Original Probability (High to Low)']
                            )

                        # Apply filters
                        df_display = df.copy()
                        if show_filter == 'Promote Only':
                            df_display = df_display[df_display['promotion_recommendation'] == 1]
                        elif show_filter == 'Eligible - No Vacancy Only':
                            df_display = df_display[df_display['promotion_recommendation'] == 2]
                        elif show_filter == 'Not Recommended Only':
                            df_display = df_display[df_display['promotion_recommendation'] == 0]

                        # Apply sorting
                        if 'Original Probability' in sort_by:
                            df_display = df_display.sort_values('original_probability', ascending=False)
                        else:
                            ascending = 'Low to High' in sort_by
                            df_display = df_display.sort_values('promotion_probability', ascending=ascending)

                        # Display columns
                        display_cols = ['department', 'job_level', 'years_in_company', 
                                       'performance_score', 'vacant_positions_in_level',
                                       'promotion_probability', 'recommendation_label']
                        
                        # Add original probability column for eligible-no vacancy cases
                        if (df_display['promotion_recommendation'] == 2).any():
                            display_cols.insert(-1, 'original_probability')

                        # Add ground truth if available
                        if has_ground_truth:
                            df_display['actual_promotion'] = df_display['promotion_received'].map({
                                1: 'Yes', 0: 'No'
                            })
                            display_cols.append('actual_promotion')

                        # Display results
                        format_dict = {
                            'promotion_probability': '{:.2%}',
                            'performance_score': '{:.2f}'
                        }
                        
                        if 'original_probability' in display_cols:
                            format_dict['original_probability'] = '{:.2%}'
                        
                        st.dataframe(
                            df_display[display_cols].style.format(format_dict),
                            width="stretch"
                        )

                        # Download buttons
                        st.markdown("---")
                        st.subheader("📥 Download Results")
                        
                        download_col1, download_col2 = st.columns(2)
                        
                        with download_col1:
                            # CSV download
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download as CSV",
                                data=csv,
                                file_name="promotion_predictions.csv",
                                mime="text/csv",
                                width="stretch"
                            )
                        
                        with download_col2:
                            # Excel download
                            excel_file = to_excel(df)
                            st.download_button(
                                label="📥 Download as Excel",
                                data=excel_file,
                                file_name="promotion_predictions.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                width="stretch"
                            )

                    except Exception as e:
                        st.error(f"Error during batch prediction: {str(e)}")
                        st.exception(e)

        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")

# -----------------------------
# Main App
# -----------------------------
def main():
    # Header
    st.title("🎯 Employee Promotion Prediction System")
    st.markdown("### Powered by Adversarial MLP - Fair AI Model")
    st.markdown("---")

    # Create tabs
    tab1, tab2 = st.tabs(["👤 Individual Prediction", "📁 Batch Prediction"])

    with tab1:
        individual_prediction_tab()

    with tab2:
        batch_prediction_tab()

    # Footer
    st.markdown("---")
    st.markdown(
        "🤖 This system uses adversarial debiasing to ensure fair predictions across departments."
    )

if __name__ == "__main__":
    main()