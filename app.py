import streamlit as st
import pandas as pd
import numpy as np
import onnxruntime as ort
import xgboost as xgb
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Subgrade order fixed globally (same as training data sorted order) ──────
SUBGRADE_ORDER = [
    'A1','A2','A3','A4','A5',
    'B1','B2','B3','B4','B5',
    'C1','C2','C3','C4','C5',
    'D1','D2','D3','D4','D5',
    'E1','E2','E3','E4','E5',
    'F1','F2','F3','F4','F5',
    'G1','G2','G3','G4','G5',
]
SUBGRADE_MAPPING = {v: i for i, v in enumerate(SUBGRADE_ORDER)}


class SimplePreprocessor:
    def __init__(self):
        self.encoders = {}
        self.is_fitted = False
        self.emp_title_mapping = {}
        self.cat_mappings = {}

    def fit_transform(self, df):
        df = df.copy()
        df['term'] = df['term'].astype(str).str.extract(r'(\d+)').astype(float)
        df['initial_list_status'] = df['initial_list_status'].map({'w': 0, 'f': 1})
        emp_map = {
            '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
            '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
            '10+ years': 10
        }
        df['emp_length'] = df['emp_length'].map(emp_map)
        grade_order = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
        df['grade'] = df['grade'].map(grade_order)
        if 'sub_grade' in df.columns:
            df['sub_grade'] = df['sub_grade'].map(SUBGRADE_MAPPING)
        cat_cols = ['verification_status', 'application_type', 'home_ownership', 'purpose']
        for col in cat_cols:
            if col in df.columns:
                unique_vals = df[col].unique()
                mapping = {val: i for i, val in enumerate(unique_vals)}
                df[col] = df[col].map(mapping)
                self.cat_mappings[col] = mapping
        if 'emp_title' in df.columns:
            top_titles = df['emp_title'].value_counts().nlargest(20).index
            df['emp_title'] = df['emp_title'].apply(lambda x: x if x in top_titles else 'Other')
            unique_titles = df['emp_title'].unique()
            title_mapping = {title: i for i, title in enumerate(unique_titles)}
            df['emp_title'] = df['emp_title'].map(title_mapping)
            self.emp_title_mapping = title_mapping
        if 'issue_d' in df.columns:
            df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
            df['issue_year'] = df['issue_d'].dt.year
            df['issue_month'] = df['issue_d'].dt.month
            df.drop(columns=['issue_d'], inplace=True)
        if 'earliest_cr_line' in df.columns:
            df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')
            df['earliest_cr_year'] = df['earliest_cr_line'].dt.year
            df.drop(columns=['earliest_cr_line'], inplace=True)
        if 'pub_rec' in df.columns:
            df['pub_rec'] = df['pub_rec'].apply(lambda x: 0 if x == 0.0 else 1)
        if 'mort_acc' in df.columns:
            df['mort_acc'] = df['mort_acc'].apply(lambda x: 0 if x == 0.0 else (1 if x >= 1.0 else x))
        if 'pub_rec_bankruptcies' in df.columns:
            df['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies'].apply(lambda x: 0 if x == 0.0 else (1 if x >= 1.0 else x))
        for col in ['title', 'address', 'zip_code', 'addr_state']:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
        self.is_fitted = True
        return df

    def transform(self, df):
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        df = df.copy()
        df['term'] = df['term'].astype(str).str.extract(r'(\d+)').astype(float)
        df['initial_list_status'] = df['initial_list_status'].map({'w': 0, 'f': 1})
        emp_map = {
            '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
            '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
            '10+ years': 10
        }
        df['emp_length'] = df['emp_length'].map(emp_map)
        grade_order = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
        df['grade'] = df['grade'].map(grade_order)

        # ── KEY FIX: use the global fixed mapping, not sort on single row ──
        if 'sub_grade' in df.columns:
            df['sub_grade'] = df['sub_grade'].map(SUBGRADE_MAPPING)

        cat_cols = ['verification_status', 'application_type', 'home_ownership', 'purpose']
        for col in cat_cols:
            if col in df.columns and col in self.cat_mappings:
                df[col] = df[col].map(self.cat_mappings[col])
        if 'emp_title' in df.columns and self.emp_title_mapping:
            df['emp_title'] = df['emp_title'].apply(
                lambda x: x if x in self.emp_title_mapping else 'Other'
            )
            df['emp_title'] = df['emp_title'].map(self.emp_title_mapping)
        if 'issue_d' in df.columns:
            df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
            df['issue_year'] = df['issue_d'].dt.year
            df['issue_month'] = df['issue_d'].dt.month
            df.drop(columns=['issue_d'], inplace=True)
        if 'earliest_cr_line' in df.columns:
            df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')
            df['earliest_cr_year'] = df['earliest_cr_line'].dt.year
            df.drop(columns=['earliest_cr_line'], inplace=True)
        if 'pub_rec' in df.columns:
            df['pub_rec'] = df['pub_rec'].apply(lambda x: 0 if x == 0.0 else 1)
        if 'mort_acc' in df.columns:
            df['mort_acc'] = df['mort_acc'].apply(lambda x: 0 if x == 0.0 else (1 if x >= 1.0 else x))
        if 'pub_rec_bankruptcies' in df.columns:
            df['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies'].apply(lambda x: 0 if x == 0.0 else (1 if x >= 1.0 else x))
        for col in ['title', 'address', 'zip_code', 'addr_state']:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
        return df


# ── Model configs ────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "Decision Tree":      {"file": "dt_model.onnx",   "type": "onnx", "accuracy": 80.4},
    "Random Forest":      {"file": "rf_model.onnx",   "type": "onnx", "accuracy": 80.6},
    "AdaBoost":           {"file": "ada_model.onnx",  "type": "onnx", "accuracy": 80.5},
    "Gradient Boosting":  {"file": "gb_model.onnx",   "type": "onnx", "accuracy": 80.7},
    "XGBoost":            {"file": "xgb_model.json",  "type": "xgb",  "accuracy": 80.8},
}

PREPROCESSOR_PATH = "dt_preprocessor.pkl"


# ── Streamlit page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Defaulters Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .prediction-box { padding: 1.5rem; border-radius: 10px; border-left: 5px solid #1f77b4; margin: 1rem 0; }
    .risk-high { background-color: #ffebee; border-left-color: #f44336; }
    .risk-low  { background-color: #e8f5e9; border-left-color: #4caf50; }
    .metric-card { background: #f8f9fa; border-radius: 8px; padding: 1rem; text-align: center; }
</style>
""", unsafe_allow_html=True)


# ── Load preprocessor (cached) ───────────────────────────────────────────────
@st.cache_resource
def load_preprocessor():
    path = Path(PREPROCESSOR_PATH)
    if not path.exists():
        st.error(f"Preprocessor not found: {PREPROCESSOR_PATH}")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


# ── Load ONNX session (cached per model file) ────────────────────────────────
@st.cache_resource
def load_onnx_session(model_file: str):
    path = Path(model_file)
    if not path.exists():
        return None
    return ort.InferenceSession(str(path))


@st.cache_resource
def load_xgb_model(model_file: str):
    path = Path(model_file)
    if not path.exists():
        return None
    booster = xgb.Booster()
    booster.load_model(str(path))
    # ── FIX: strip feature names saved during training so inference
    #         accepts a plain numpy array without name validation ──────
    booster.feature_names = None
    booster.feature_types = None
    return booster


def get_onnx_outputs(session, input_array: np.ndarray):
    """
    Safely extract (prediction, prob_default) from any sklearn ONNX model.
    Handles all three output formats skl2onnx produces.
    """
    input_name = session.get_inputs()[0].name
    results = session.run(None, {input_name: input_array})

    prediction = int(results[0][0])

    prob_default = None

    if len(results) > 1:
        prob_output = results[1]

        # Format 1: list of dicts  [{0: 0.82, 1: 0.18}]  ← most sklearn models
        if isinstance(prob_output, list) and len(prob_output) > 0:
            first = prob_output[0]
            if isinstance(first, dict):
                prob_default = float(first.get(0, first.get(0.0, None)))

        # Format 2: numpy array shape (1, 2)  ← some tree models
        elif isinstance(prob_output, np.ndarray):
            arr = np.squeeze(prob_output)          # flatten to 1D
            if arr.ndim == 1 and len(arr) == 2:
                # arr[0] = prob class 0 (default), arr[1] = prob class 1 (repay)
                # Only trust it if values are valid probabilities
                if 0.0 <= float(arr[0]) <= 1.0 and 0.0 <= float(arr[1]) <= 1.0:
                    prob_default = float(arr[0])
            elif arr.ndim == 0:
                # Scalar — raw score, NOT a probability, ignore it
                prob_default = None

    # If probability extraction fails, stay neutral instead of
    # assigning a favorable/unfavorable probability from the prediction.
    if prob_default is None or not (0.0 <= prob_default <= 1.0):
        prob_default = 0.5

    return prediction, prob_default


def preprocess_input(preprocessor, loan_data: dict) -> np.ndarray | None:
    """Transform raw dict → float32 numpy array ready for inference."""
    try:
        df = pd.DataFrame([loan_data])
        processed = preprocessor.transform(df)
        if 'loan_status' in processed.columns:
            processed = processed.drop(columns=['loan_status'])
        # .values strips all pandas column metadata — safe for both ONNX and XGBoost
        return processed.values.astype(np.float32)
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return None


# ── Sidebar input form ───────────────────────────────────────────────────────
def create_input_form():
    st.sidebar.header("📊 Loan Application Details")

    st.sidebar.subheader("🤖 Model Selection")
    model_name = st.sidebar.selectbox("Choose Model", list(MODEL_CONFIGS.keys()))

    st.sidebar.subheader("💰 Loan Information")
    loan_amnt  = st.sidebar.number_input("Loan Amount ($)", 1000, 500000, 15000, 1000)
    term       = st.sidebar.selectbox("Loan Term", ["36 months", "60 months"])
    int_rate   = st.sidebar.slider("Interest Rate (%)", 5.0, 30.0, 12.0, 0.1)
    installment = st.sidebar.number_input("Monthly Installment ($)", 100, 20000, 500, 50)

    st.sidebar.subheader("👤 Borrower Information")
    annual_inc    = st.sidebar.number_input("Annual Income ($)", 10000, 1000000, 60000, 5000)
    emp_length    = st.sidebar.selectbox("Employment Length",
                        ["< 1 year","1 year","2 years","3 years","4 years",
                         "5 years","6 years","7 years","8 years","9 years","10+ years"])
    home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT","MORTGAGE","OWN","OTHER"])

    st.sidebar.subheader("💳 Credit Information")
    dti       = st.sidebar.slider("Debt-to-Income Ratio", 0.0, 100.0, 20.0, 0.5)
    open_acc  = st.sidebar.number_input("Open Credit Lines", 0, 50, 8, 1)
    total_acc = st.sidebar.number_input("Total Credit Lines", 0, 100, 15, 1)
    revol_bal  = st.sidebar.number_input("Revolving Balance ($)", 0, 100000, 8000, 1000)
    revol_util = st.sidebar.slider("Revolving Utilization (%)", 0.0, 150.0, 65.0, 1.0)

    st.sidebar.subheader("📋 Loan Details")
    purpose = st.sidebar.selectbox("Loan Purpose",
                ["debt_consolidation","credit_card","home_improvement","major_purchase",
                 "medical","vacation","wedding","car","moving","house","other"])
    verification_status = st.sidebar.selectbox("Income Verification",
                              ["Verified","Not Verified","Source Verified"])
    application_type = st.sidebar.selectbox("Application Type", ["Individual", "Joint App"])

    st.sidebar.subheader("📈 Grade")
    grade = st.sidebar.selectbox("Loan Grade", ["A","B","C","D","E","F","G"])
    sub_grade = st.sidebar.selectbox("Sub Grade", SUBGRADE_ORDER)

    loan_data = {
        'loan_amnt': loan_amnt, 'term': term, 'int_rate': int_rate,
        'installment': installment, 'grade': grade, 'sub_grade': sub_grade,
        'emp_title': 'Other',          # mapped to 'Other' bucket in preprocessor
        'emp_length': emp_length, 'home_ownership': home_ownership,
        'annual_inc': annual_inc, 'verification_status': verification_status,
        'purpose': purpose, 'dti': dti, 'open_acc': open_acc,
        'pub_rec': 0.0, 'revol_bal': revol_bal, 'revol_util': revol_util,
        'total_acc': total_acc, 'initial_list_status': 'w',
        'application_type': application_type,
        'mort_acc': 1.0, 'pub_rec_bankruptcies': 0.0,
        'issue_d': 'Jan-2024', 'earliest_cr_line': 'Jan-2010',
    }
    return loan_data, model_name


# ── Result display ───────────────────────────────────────────────────────────
def display_prediction(prediction: int, prob_default: float, model_name: str, loan_data: dict):
    confidence_pct = prob_default * 100          # probability of DEFAULT
    repay_pct      = (1 - prob_default) * 100    # probability of REPAYMENT

    st.markdown('<h2 class="main-header">🎯 Prediction Result</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model Used", model_name)
    col2.metric("Prediction", "⚠️ DEFAULT" if prediction == 0 else "✅ REPAY")
    col3.metric("Default Risk", f"{confidence_pct:.1f}%")
    col4.metric("Repay Probability", f"{repay_pct:.1f}%")

    if prediction == 0:
        st.markdown(f"""
        <div class="prediction-box risk-high">
            <h3>🚨 HIGH RISK — Likely to Default</h3>
            <p><strong>Default Probability:</strong> {confidence_pct:.1f}%</p>
            <p><strong>Analysis:</strong> This application shows characteristics associated with higher default risk.</p>
            <p><strong>Recommendation:</strong> Consider additional verification, higher interest rate, or reduced loan amount.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box risk-low">
            <h3>✅ LOW RISK — Likely to Repay</h3>
            <p><strong>Repayment Probability:</strong> {repay_pct:.1f}%</p>
            <p><strong>Analysis:</strong> This application meets standard lending criteria.</p>
            <p><strong>Recommendation:</strong> Application can proceed through normal approval process.</p>
        </div>""", unsafe_allow_html=True)

    # Probability gauge
    st.subheader("📊 Risk Breakdown")
    risk_df = pd.DataFrame({
        "Outcome":      ["Will Repay ✅", "Will Default ⚠️"],
        "Probability":  [round(repay_pct, 2), round(confidence_pct, 2)]
    })
    st.bar_chart(risk_df.set_index("Outcome"))

    # Summary
    st.subheader("📋 Application Summary")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Loan Details:**")
        st.write(f"- Amount: ${loan_data['loan_amnt']:,}")
        st.write(f"- Term: {loan_data['term']}")
        st.write(f"- Interest Rate: {loan_data['int_rate']}%")
        st.write(f"- Monthly Payment: ${loan_data['installment']:,}")
    with c2:
        st.write("**Borrower Details:**")
        st.write(f"- Annual Income: ${loan_data['annual_inc']:,}")
        st.write(f"- Employment: {loan_data['emp_length']}")
        st.write(f"- Home Ownership: {loan_data['home_ownership']}")
        st.write(f"- DTI Ratio: {loan_data['dti']}%")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    st.markdown('<h1 class="main-header">🏦 Loan Default Predictor</h1>', unsafe_allow_html=True)
    st.markdown("##### AI-powered risk assessment using multiple trained models")

    preprocessor = load_preprocessor()
    if preprocessor is None:
        st.stop()

    loan_data, model_name = create_input_form()

    cfg = MODEL_CONFIGS[model_name]

    if cfg["type"] == "xgb":
        model = load_xgb_model(cfg["file"])
        if model is None:
            st.warning(f"Model file **{cfg['file']}** not found.")
            st.stop()
    else:
        model = load_onnx_session(cfg["file"])
        if model is None:
            st.warning(f"Model file **{cfg['file']}** not found.")
            st.stop()

    if st.sidebar.button("🚀 Predict Loan Risk", type="primary", use_container_width=True):
        with st.spinner("Analysing loan application…"):
            input_array = preprocess_input(preprocessor, loan_data)
            if input_array is not None:
                try:
                    if cfg["type"] == "xgb":
                        dmatrix = xgb.DMatrix(input_array, feature_names=None)
                        prob_repay   = float(np.clip(model.predict(dmatrix)[0], 0.0, 1.0))
                        prob_default = 1.0 - prob_repay

                        # Small decision-threshold adjustment to avoid an
                        # overly approval-heavy displayed result.
                        prediction = 1 if prob_repay >= 0.60 else 0
                    else:
                        prediction, prob_default = get_onnx_outputs(model, input_array)
                        prob_default = float(np.clip(prob_default, 0.0, 1.0))  # safety clamp

                    display_prediction(prediction, prob_default, model_name, loan_data)

                except Exception as e:
                    st.error(f"Inference error: {e}")
                    st.exception(e)   # shows full traceback in UI during debugging

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Model Accuracies")
    for name, cfg in MODEL_CONFIGS.items():
        st.sidebar.write(f"- **{name}:** {cfg['accuracy']}%")

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;color:#888;'>
        Built with ❤️ using Streamlit · ONNX Runtime · Lending Club Dataset
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
