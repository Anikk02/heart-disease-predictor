# heart_disease_xgboost_app.py - SECURE VERSION WITH HIDDEN CONFIG
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import traceback
import os
import tempfile
from huggingface_hub import hf_hub_download, login

# --- Page config: fullscreen style ---
st.set_page_config(
    page_title="Heart Disease Predictor", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="❤️"
)

# CSS (Keep ALL your CSS as is)
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(90deg, #0b6bff, #00d4ff);
        padding: 1.5rem;
        border-radius: 0px;
        margin-bottom: 2rem;
        color: white;
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        border: 3px solid #4a5568;
    }
    .prediction-high-risk {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff4757 100%);
    }
    .prediction-low-risk {
        background: linear-gradient(135deg, #2ed573 0%, #1e90ff 100%);
    }
    .parameter-analysis-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #0b6bff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .parameter-item {
        padding: 0.5rem 0;
        border-bottom: 1px solid #e9ecef;
    }
    .parameter-item:last-child {
        border-bottom: none;
    }
    .parameter-name {
        font-weight: 600;
        color: #2c3e50;
    }
    .parameter-value {
        font-weight: 700;
        color: #0b6bff;
    }
    .parameter-status-normal {
        color: #2ed573;
        font-weight: 500;
    }
    .parameter-status-warning {
        color: #ffa502;
        font-weight: 500;
    }
    .parameter-status-danger {
        color: #ff4757;
        font-weight: 500;
    }
    .parameter-explanation {
        color: #666;
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }
    .recommendation-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 5px solid #0b6bff;
    }
    .recommendation-item {
        margin: 0.75rem 0;
        padding-left: 1.5rem;
        position: relative;
    }
    .recommendation-item:before {
        content: "•";
        color: #0b6bff;
        font-size: 1.5rem;
        position: absolute;
        left: 0;
        top: -2px;
    }
    .recommendation-category {
        color: #2c3e50;
        font-weight: 600;
        margin: 1rem 0 0.5rem 0;
        padding-bottom: 0.25rem;
        border-bottom: 2px solid #e9ecef;
    }
    .warning-box {
        background: linear-gradient(90deg, #ffefef, #ffeaea);
        border-left: 5px solid #ff6b6b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #0b6bff, #00d4ff);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(11, 107, 255, 0.3);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
    /* input field alignment */
    .stNumberInput, .stSelectbox, .stRadio, .stSlider {
        margin-bottom: 1rem;
    }
    /* column spacing */
    .stColumn {
        padding: 0 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Feature Importance Weights ---
FEATURE_IMPORTANCE = {
    'ca': 0.1616,
    'thalach': 0.1597,
    'thal': 0.1155,
    'exang': 0.1115,
    'age': 0.1032,
    'oldpeak': 0.0928,
    'cp': 0.0887,
    'fbs': 0.0513,
    'slope': 0.0346,
    'restecg': 0.0291,
    'chol': 0.0185,
    'trestbps': 0.0181,
    'sex': 0.0153
}

# --- HIDDEN CONFIGURATION (NOT IN SIDEBAR) ---
# Load from environment variables or secrets
class HiddenConfig:
    # Repository details - loaded from environment/secrets
    REPO_ID = os.getenv("HF_REPO_ID", "")
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    
    # File names - fixed (not editable by users)
    MODEL_FILENAME = "xgboost_model.pkl"
    SCALER_FILENAME = "scaler.pkl"
    FEATURES_FILENAME = "feature_order.pkl"
    
    # Cache setting - fixed
    USE_CACHE = True
    
    # Threshold - can be set as default or from env
    DEFAULT_THRESHOLD = 0.627
    
    @classmethod
    def load_from_secrets(cls):
        """Load configuration from Streamlit secrets if available"""
        if hasattr(st, 'secrets'):
            if 'HF_TOKEN' in st.secrets:
                cls.HF_TOKEN = st.secrets['HF_TOKEN']
            if 'HF_REPO_ID' in st.secrets:
                cls.REPO_ID = st.secrets['HF_REPO_ID']
            if 'MODEL_THRESHOLD' in st.secrets:
                try:
                    cls.DEFAULT_THRESHOLD = float(st.secrets['MODEL_THRESHOLD'])
                except:
                    pass

# Initialize hidden configuration
config = HiddenConfig()
config.load_from_secrets()

# --- HEADER ---
st.markdown(
    """
    <div class="main-header">
        <h1 style="margin:0; color:white">❤️ Heart Disease Risk Assessment</h1>
        <h3 style="margin:0; color:white; font-weight:300">Powered by XGBoost Model (94.6% Accuracy)</h3>
        <p style="margin:0.5rem 0 0 0; color:white; opacity:0.9">
        Top features: <b>Number of vessels (ca)</b>, <b>Max heart rate (thalach)</b>, <b>Thalassemia (thal)</b>, 
        <b>Exercise angina (exang)</b>, <b>Age</b>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption("⚠️ **Important**: This tool provides risk estimates for educational purposes only. It is NOT a medical diagnosis. Always consult healthcare professionals for medical advice.")

# --- SIDEBAR: CLEANED VERSION (NO MODEL CONFIG) ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Only show user-facing settings, NOT model configuration
    st.subheader("🎯 Risk Threshold")
    threshold = st.slider(
        "Probability threshold for 'High Risk' classification",
        min_value=0.0,
        max_value=1.0,
        value=config.DEFAULT_THRESHOLD,
        step=0.01,
        help="Higher = more conservative (fewer false positives), Lower = more sensitive (catches more cases)"
    )
    
    st.info(f"""
    **Threshold {threshold:.3f}**: 
    - Precision: ~88%
    - Recall: ~91%
    - F1 Score: ~89%
    """)
    
    st.subheader("📈 Feature Importance")
    feature_to_view = st.selectbox("View details for:", list(FEATURE_IMPORTANCE.keys()))
    importance = FEATURE_IMPORTANCE[feature_to_view]
    st.progress(importance, text=f"{feature_to_view}: {importance:.1%} importance")

    if feature_to_view in ['ca', 'thalach', 'thal', 'exang', 'oldpeak']:
        feature_descriptions = {
            'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
            'thalach': 'Maximum heart rate achieved during exercise',
            'thal': 'Thalassemia blood disorder results',
            'exang': 'Exercise-induced angina (chest pain)',
            'oldpeak': 'ST depression induced by exercise'
        }
        st.info(f"**Clinical significance**: {feature_descriptions[feature_to_view]}")
    
    # Add model status indicator (optional)
    st.subheader("🔍 Model Status")
    if 'model_loaded' in st.session_state and st.session_state.model_loaded:
        st.success("✅ Model ready")
    else:
        st.info("🔄 Model will load on prediction")

# --- Function to load model from Hugging Face (UPDATED TO USE HIDDEN CONFIG) ---
@st.cache_resource(show_spinner=False)
def load_model_from_huggingface():
    """Load model artifacts from Hugging Face Hub using hidden configuration"""
    
    try:
        # Validate token for private repository
        if not config.HF_TOKEN or config.HF_TOKEN.strip() == "":
            raise ValueError("Hugging Face token is required for private repositories")
        
        # Login to Hugging Face with token
        login(token=config.HF_TOKEN)
        
        cache_dir = None
        if config.USE_CACHE:
            cache_dir = tempfile.mkdtemp(prefix="heart_model_")
        
        # Download model files with authentication
        model_path = hf_hub_download(
            repo_id=config.REPO_ID,
            filename=config.MODEL_FILENAME,
            token=config.HF_TOKEN,
            cache_dir=cache_dir,
            force_download=False
        )
        
        scaler_path = hf_hub_download(
            repo_id=config.REPO_ID,
            filename=config.SCALER_FILENAME,
            token=config.HF_TOKEN,
            cache_dir=cache_dir,
            force_download=False
        )
        
        features_path = hf_hub_download(
            repo_id=config.REPO_ID,
            filename=config.FEATURES_FILENAME,
            token=config.HF_TOKEN,
            cache_dir=cache_dir,
            force_download=False
        )
        
        # Load the downloaded files
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_order = joblib.load(features_path)
        
        return model, scaler, feature_order
        
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            raise PermissionError("Authentication failed. Please check configuration.")
        elif "404" in error_msg or "Not Found" in error_msg:
            raise FileNotFoundError(f"Model resources not found.")
        else:
            raise Exception(f"Error loading model: {error_msg}")

# --- MAIN CONTENT: Input Section ---
# KEEP ALL YOUR EXISTING INPUT CODE EXACTLY AS IS
st.header("📝 Patient Clinical Parameters")

# Create three columns for better distribution
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics & Vital Signs")
    
    age = st.number_input(
        "**Age** (years) 👴", 
        min_value=1, 
        max_value=120, 
        value=55, 
        step=1,
        help="Risk increases significantly after 55 years"
    )
    
    sex = st.radio(
        "**Sex** 👥", 
        options=["Male", "Female"], 
        index=0,
        help="Males generally have higher risk of heart disease"
    )
    
    trestbps = st.number_input(
        "**Resting Blood Pressure** (trestbps) 🩺", 
        min_value=60, 
        max_value=250, 
        value=130, 
        step=1,
        help="Optimal: <120 mmHg, High: ≥140 mmHg"
    )
    
    chol = st.number_input(
        "**Serum Cholesterol** (chol) 🧪", 
        min_value=80, 
        max_value=600, 
        value=200, 
        step=1,
        help="Optimal: <200 mg/dL, High: ≥240 mg/dL"
    )

with col2:
    st.subheader("Symptoms & Pain")
    
    # Chest pain options
    cp_options = [
        "Typical angina (0)",
        "Atypical angina (1)", 
        "Non-anginal pain (2)",
        "Asymptomatic (3)"
    ]
    
    cp = st.selectbox(
        "**Chest Pain Type** (cp) 💔", 
        options=cp_options,
        index=3,
        help="Asymptomatic and typical angina are higher risk indicators"
    )
    
    exang = st.radio(
        "**Exercise-Induced Angina** (exang) 🏃", 
        options=["No (0)", "Yes (1)"], 
        index=0,
        help="Chest pain during exercise is a strong predictor of coronary artery disease"
    )
    
    fbs = st.radio(
        "**Fasting Blood Sugar >120 mg/dL** (fbs) 🍬", 
        options=["No (0)", "Yes (1)"], 
        index=0,
        help="Elevated fasting glucose indicates diabetes risk"
    )

with col3:
    st.subheader("Test Results")
    
    ca = st.slider(
        "**Number of Major Vessels** (ca) 🫀", 
        min_value=0, 
        max_value=3, 
        value=0,
        step=1,
        help="0 = No major vessels colored, 3 = Three major vessels colored (higher risk)"
    )
    
    thalach = st.number_input(
        "**Maximum Heart Rate Achieved** (thalach) 💓", 
        min_value=50, 
        max_value=250, 
        value=150, 
        step=1,
        help="Lower maximum heart rate suggests poor cardiovascular fitness (higher risk)"
    )
    
    oldpeak = st.number_input(
        "**ST Depression** (oldpeak) 📉", 
        min_value=0.0, 
        max_value=6.0, 
        value=1.0, 
        step=0.1,
        format="%.1f",
        help="ST segment depression during exercise (>2.0 indicates higher risk)"
    )

# Additional column for remaining inputs
col4, col5, col6 = st.columns(3)

with col4:
    thal_options = [
        "Normal (0)",
        "Fixed defect (1)", 
        "Reversible defect (2)"
    ]
    
    thal = st.selectbox(
        "**Thalassemia** (thal) 🩸", 
        options=thal_options,
        index=0,
        help="Fixed and reversible defects indicate blood flow issues to the heart"
    )

with col5:
    restecg_options = [
        "Normal (0)",
        "ST-T wave abnormality (1)",
        "Probable/definite LVH (2)"
    ]
    
    restecg = st.selectbox(
        "**Resting ECG Results** (restecg) 📊", 
        options=restecg_options,
        index=0,
        help="LVH = Left Ventricular Hypertrophy"
    )

with col6:
    slope_options = [
        "Upsloping (1)",
        "Flat (2)", 
        "Downsloping (3)"
    ]
    
    slope = st.selectbox(
        "**Slope of Peak Exercise ST Segment** (slope) ↗️", 
        options=slope_options,
        index=0,
        help="Upsloping = Normal, Downsloping = Abnormal (higher risk)"
    )

# --- Converting inputs to numeric values ---
input_mapping = {
    'cp': {
        "Typical angina (0)": 0, 
        "Atypical angina (1)": 1, 
        "Non-anginal pain (2)": 2, 
        "Asymptomatic (3)": 3
    },
    'thal': {
        "Normal (0)": 0, 
        "Fixed defect (1)": 1, 
        "Reversible defect (2)": 2
    },
    'slope': {
        "Upsloping (1)": 1, 
        "Flat (2)": 2, 
        "Downsloping (3)": 3
    },
    'restecg': {
        "Normal (0)": 0, 
        "ST-T wave abnormality (1)": 1, 
        "Probable/definite LVH (2)": 2
    }
}

try:
    # Convert inputs
    age_v = int(age)
    sex_v = 1 if sex == "Male" else 0
    cp_v = input_mapping['cp'][cp]
    exang_v = 1 if exang.startswith("Yes") else 0
    thal_v = input_mapping['thal'][thal]
    ca_v = int(ca)
    thalach_v = int(thalach)
    oldpeak_v = float(oldpeak)
    trest_v = int(trestbps)
    chol_v = int(chol)
    fbs_v = 1 if fbs.startswith("Yes") else 0
    restecg_v = input_mapping['restecg'][restecg]
    slope_v = input_mapping['slope'][slope]
    
    # Create features array
    features_array = np.array([[
        age_v, sex_v, cp_v, trest_v, chol_v, fbs_v, restecg_v,
        thalach_v, exang_v, oldpeak_v, slope_v, ca_v, thal_v
    ]], dtype=float)
    
    feature_names = ['age','sex','cp','trestbps','chol','fbs','restecg',
                    'thalach','exang','oldpeak','slope','ca','thal']
    
    features_df = pd.DataFrame(features_array, columns=feature_names)
    
except Exception as e:
    st.error(f"Error processing inputs: {str(e)}")
    st.code(traceback.format_exc())
    st.stop()

# --- Display Input Summary ---
st.markdown("---")
st.subheader("📋 Input Summary")

# Create a summary table
summary_data = {
    "Feature": ["Age", "Sex", "Chest Pain", "Exercise Angina", "Thalassemia", 
                "Major Vessels", "Max HR", "ST Depression", "Resting BP", 
                "Cholesterol", "Fasting Sugar", "Resting ECG", "ST Slope"],
    "Value": [
        f"{age_v} years",
        "Male" if sex_v == 1 else "Female",
        ["Typical angina", "Atypical angina", "Non-anginal", "Asymptomatic"][cp_v],
        "Yes" if exang_v == 1 else "No",
        ["Normal", "Fixed defect", "Reversible defect"][thal_v],
        f"{ca_v}",
        f"{thalach_v} bpm",
        f"{oldpeak_v:.1f}",
        f"{trest_v} mmHg",
        f"{chol_v} mg/dL",
        ">120 mg/dL" if fbs_v == 1 else "Normal",
        ["Normal", "ST-T abnormal", "LVH"][restecg_v],
        ["Upsloping", "Flat", "Downsloping"][slope_v - 1]
    ]
}

summary_df = pd.DataFrame(summary_data)

# Display in a table format
st.dataframe(
    summary_df,
    column_config={
        "Feature": st.column_config.TextColumn("Clinical Feature", width="medium"),
        "Value": st.column_config.TextColumn("Patient Value", width="medium"),
    },
    hide_index=True,
    use_container_width=True
)

# --- PARAMETER ANALYSIS SECTION ---
# KEEP ALL YOUR EXISTING PARAMETER ANALYSIS CODE EXACTLY AS IS
st.markdown("---")
st.subheader("🔍 Parameter Analysis")

# Define parameter analysis function
def analyze_parameter(param_name, param_value, unit=""):
    """Generate analysis for a single parameter"""
    
    # Get status and explanation based on parameter
    status = "within common range"
    explanation = ""
    status_class = "parameter-status-normal"
    is_warning = False
    is_danger = False
    
    if param_name == "Resting BP":
        value = param_value
        if value >= 140:
            status = "High (≥140 mmHg)"
            status_class = "parameter-status-danger"
            explanation = "High blood pressure increases heart workload. (Value ≥140 → may increase cardiac risk)"
            is_danger = True
        elif value >= 120:
            status = "Elevated (≥120 mmHg)"
            status_class = "parameter-status-warning"
            explanation = "High blood pressure increases heart workload. (Value >120 → may increase cardiac risk)"
            is_warning = True
        else:
            status = "Normal (<120 mmHg)"
            explanation = "Optimal blood pressure level"
    
    elif param_name == "Cholesterol":
        value = param_value
        if value >= 240:
            status = "High (≥240 mg/dL)"
            status_class = "parameter-status-danger"
            explanation = "High cholesterol increases plaque buildup in arteries"
            is_danger = True
        elif value >= 200:
            status = "Borderline High (≥200 mg/dL)"
            status_class = "parameter-status-warning"
            explanation = "Moderately elevated cholesterol level"
            is_warning = True
        else:
            status = "Desirable (<200 mg/dL)"
            explanation = "Within optimal range"
    
    elif param_name == "Fasting blood sugar >120":
        value = param_value
        if value == 1:
            status = "Elevated (≥120 mg/dL)"
            status_class = "parameter-status-warning"
            explanation = "May indicate prediabetes or diabetes risk"
            is_warning = True
        else:
            status = "Normal (<120 mg/dL)"
            explanation = "Within normal fasting glucose range"
    
    elif param_name == "Max heart rate":
        value = param_value
        if value < 120:
            status = "Low (<120 bpm)"
            status_class = "parameter-status-danger"
            explanation = "Low maximum heart rate may indicate poor cardiovascular fitness"
            is_danger = True
        elif value < 140:
            status = "Below optimal (<140 bpm)"
            status_class = "parameter-status-warning"
            explanation = "Moderate cardiovascular fitness"
            is_warning = True
        else:
            status = "Optimal (≥140 bpm)"
            explanation = "Good cardiovascular fitness"
    
    elif param_name == "ST depression (oldpeak)":
        value = param_value
        if value >= 2.0:
            status = "Significant (≥2.0)"
            status_class = "parameter-status-danger"
            explanation = "May indicate myocardial ischemia during exercise"
            is_danger = True
        elif value >= 1.0:
            status = "Mild (≥1.0)"
            status_class = "parameter-status-warning"
            explanation = "Borderline ST depression"
            is_warning = True
        else:
            status = "Normal (<1.0)"
            explanation = "No significant ST depression"
    
    elif param_name == "Major vessels (ca)":
        value = param_value
        if value >= 2:
            status = f"Multiple ({int(value)} vessels)"
            status_class = "parameter-status-danger"
            explanation = f"{int(value)} major vessels colored indicates significant CAD"
            is_danger = True
        elif value == 1:
            status = "Single vessel"
            status_class = "parameter-status-warning"
            explanation = "One major vessel colored, mild coronary artery disease"
            is_warning = True
        else:
            status = "No vessels (0)"
            explanation = "No major vessels colored, normal finding"
    
    elif param_name == "Age":
        value = param_value
        if value >= 65:
            status = "Senior (≥65 years)"
            status_class = "parameter-status-warning"
            explanation = "Age is a significant risk factor for heart disease"
            is_warning = True
        elif value >= 55:
            status = "Middle-aged (≥55 years)"
            status_class = "parameter-status-warning"
            explanation = "Moderate age-related risk increase"
            is_warning = True
        else:
            status = "Younger (<55 years)"
            explanation = "Lower age-related risk"
    
    elif param_name == "Exercise angina":
        value = param_value
        if value == 1:
            status = "Present"
            status_class = "parameter-status-danger"
            explanation = "Exercise-induced chest pain is a classic symptom of CAD"
            is_danger = True
        else:
            status = "Absent"
            explanation = "No exercise-induced chest pain reported"
    
    # Add unit to value if provided
    display_value = f"{param_value}"
    if unit:
        display_value += f" {unit}"
    
    return {
        "name": param_name,
        "display_value": display_value,
        "status": status,
        "status_class": status_class,
        "explanation": explanation,
        "is_warning": is_warning,
        "is_danger": is_danger,
        "value": param_value
    }

# Define parameters to analyze
parameters_to_analyze = [
    ("Resting BP", trest_v, "mmHg"),
    ("Cholesterol", chol_v, "mg/dL"),
    ("Fasting blood sugar >120", fbs_v, ""),
    ("Max heart rate", thalach_v, "bpm"),
    ("ST depression (oldpeak)", oldpeak_v, ""),
    ("Major vessels (ca)", ca_v, ""),
    ("Age", age_v, "years"),
    ("Exercise angina", exang_v, ""),
]

# Store parameter analyses for later use
parameter_analyses = []
high_risk_params = []

# Display parameter analysis in a box
st.markdown('<div class="parameter-analysis-box">', unsafe_allow_html=True)

for param_name, param_value, unit in parameters_to_analyze:
    analysis = analyze_parameter(param_name, param_value, unit)
    parameter_analyses.append(analysis)
    
    if analysis["is_warning"] or analysis["is_danger"]:
        high_risk_params.append(analysis)
    
    st.markdown(f"""
    <div class="parameter-item">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <span class="parameter-name">{analysis['name']}:</span>
                <span class="parameter-value"> {analysis['display_value']}</span>
            </div>
            <span class="{analysis['status_class']}">{analysis['status']}</span>
        </div>
        {f'<div class="parameter-explanation">{analysis["explanation"]}</div>' if analysis["explanation"] else ''}
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# feature importance note
st.caption("💡 **Note**: The most important features for this model are: Number of vessels (ca), Max heart rate (thalach), Thalassemia (thal), Exercise angina (exang), and Age.")

# --- PREDICTION BUTTON ---
st.markdown("---")
col_pred1, col_pred2, col_pred3 = st.columns([1, 2, 1])

with col_pred2:
    predict_button = st.button(
        "🚀 PREDICT HEART DISEASE RISK",
        use_container_width=True,
        type="primary"
    )

# --- PREDICTION LOGIC ---
if predict_button:
    with st.spinner("🔍 Loading clinical prediction model..."):
        try:
            # Load model using hidden configuration
            model, scaler, feature_order = load_model_from_huggingface()
            
            # Update session state
            st.session_state.model_loaded = True
            
            # Scale features
            X_scaled = scaler.transform(features_array)
            
            # Make prediction
            probability = float(model.predict_proba(X_scaled)[0, 1])
            prediction = 1 if probability >= threshold else 0
            
            # Clear spinner
            st.success("✅ Prediction completed!")
            
            # --- DISPLAY RESULTS ---
            # KEEP ALL YOUR EXISTING RESULT DISPLAY CODE EXACTLY AS IS
            st.markdown("---")
            
            # Determine if it's high or low risk
            is_high_risk = probability >= threshold
            
            # Use consistent purple gradient for the main card
            prediction_card_class = "prediction-high-risk" if is_high_risk else "prediction-low-risk"
            
            # Result Card - SIMPLE AND CONSISTENT DESIGN
            st.markdown(f"""
            <div class="prediction-card {prediction_card_class}">
                <h1 style="margin:0; font-size:42px; font-weight:700;">
                    {"VERY HIGH RISK" if probability >= 0.8 else "HIGH RISK" if is_high_risk else "LOW RISK"}
                </h1>
                <h1 style="margin:20px 0; font-size:64px; font-weight:900;">{probability:.1%}</h1>
                <p style="margin:0; font-size:18px; opacity:0.9">
                Probability of heart disease based on XGBoost model<br>
                Threshold for high risk: {threshold:.0%}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Simple metrics below the card
            col_dec1, col_dec2 = st.columns(2)
            
            with col_dec1:
                st.metric(
                    label="Model Prediction",
                    value="HIGH RISK" if is_high_risk else "LOW RISK",
                    delta=f"{'≥' if is_high_risk else '<'} {threshold:.0%} threshold",
                    delta_color="inverse" if is_high_risk else "normal"
                )
            
            with col_dec2:
                confidence = abs(probability - 0.5) * 2
                st.metric(
                    label="Model Confidence",
                    value=f"{confidence:.1%}",
                    delta="High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"
                )
            
            # --- SPECIFIC CLINICAL RECOMMENDATIONS ---
            st.markdown("---")
            st.subheader("🎯 Specific Clinical Recommendations")
            
            st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
            
            if high_risk_params:
                # Show recommendations for high risk parameters
                st.write("**Based on your elevated parameters:**")
                
                for param in high_risk_params:
                    param_name = param["name"]
                    param_value = param["value"]
                    param_status = param["status"]
                    
                    st.markdown(f'<div class="recommendation-category">{param_name} ({param_status})</div>', unsafe_allow_html=True)
                    
                    if param_name == "Resting BP":
                        if param_value >= 140:
                            st.markdown("""
                            <div class="recommendation-item">Immediate BP evaluation by physician</div>
                            <div class="recommendation-item">Start BP monitoring 2x daily</div>
                            <div class="recommendation-item">Consider DASH diet (low sodium, high potassium)</div>
                            <div class="recommendation-item">Limit alcohol to 1 drink/day max</div>
                            <div class="recommendation-item">Medication may be needed (consult doctor)</div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="recommendation-item">Lifestyle modifications to reduce BP</div>
                            <div class="recommendation-item">Increase physical activity (30 min/day)</div>
                            <div class="recommendation-item">Reduce sodium intake below 2300mg/day</div>
                            <div class="recommendation-item">Monitor BP weekly</div>
                            """, unsafe_allow_html=True)
                    
                    elif param_name == "Cholesterol":
                        if param_value >= 240:
                            st.markdown("""
                            <div class="recommendation-item">Statin therapy likely needed (consult doctor)</div>
                            <div class="recommendation-item">Reduce saturated fats to <7% of calories</div>
                            <div class="recommendation-item">Increase soluble fiber (oats, beans, apples)</div>
                            <div class="recommendation-item">Consider plant sterols/stanols</div>
                            <div class="recommendation-item">Repeat lipid panel in 3 months</div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="recommendation-item">Reduce dietary cholesterol (<300mg/day)</div>
                            <div class="recommendation-item">Increase omega-3 fatty acids (fish, flaxseed)</div>
                            <div class="recommendation-item">Regular aerobic exercise (150 min/week)</div>
                            <div class="recommendation-item">Maintain healthy weight</div>
                            """, unsafe_allow_html=True)
                    
                    elif param_name == "Fasting blood sugar >120":
                        st.markdown("""
                        <div class="recommendation-item">Diabetes screening (HbA1c test)</div>
                        <div class="recommendation-item">Reduce sugar and refined carb intake</div>
                        <div class="recommendation-item">Increase physical activity</div>
                        <div class="recommendation-item">Monitor fasting glucose monthly</div>
                        <div class="recommendation-item">Consider metformin if pre-diabetic (consult doctor)</div>
                        """, unsafe_allow_html=True)
                    
                    elif param_name == "Max heart rate":
                        if param_value < 120:
                            st.markdown("""
                            <div class="recommendation-item">Cardiac stress test recommended</div>
                            <div class="recommendation-item">Start supervised cardiac rehab program</div>
                            <div class="recommendation-item">Gradual increase in exercise intensity</div>
                            <div class="recommendation-item">Check for beta-blocker medication effects</div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="recommendation-item">Increase aerobic exercise frequency</div>
                            <div class="recommendation-item">Interval training to improve cardio fitness</div>
                            <div class="recommendation-item">Target heart rate zone training</div>
                            <div class="recommendation-item">Regular cardiovascular conditioning</div>
                            """, unsafe_allow_html=True)
                    
                    elif param_name == "ST depression (oldpeak)":
                        if param_value >= 2.0:
                            st.markdown("""
                            <div class="recommendation-item">**Urgent cardiology referral needed**</div>
                            <div class="recommendation-item">Nuclear stress test or stress echo</div>
                            <div class="recommendation-item">Consider coronary angiography</div>
                            <div class="recommendation-item">Avoid strenuous exercise until evaluated</div>
                            <div class="recommendation-item">Anti-anginal medications may be needed</div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="recommendation-item">Follow-up stress test in 6-12 months</div>
                            <div class="recommendation-item">Monitor for chest pain with exertion</div>
                            <div class="recommendation-item">Cardiology consultation recommended</div>
                            """, unsafe_allow_html=True)
                    
                    elif param_name == "Major vessels (ca)":
                        if param_value >= 2:
                            st.markdown("""
                            <div class="recommendation-item">**Immediate cardiology consultation**</div>
                            <div class="recommendation-item">Coronary angiography likely indicated</div>
                            <div class="recommendation-item">Consider PCI or CABG evaluation</div>
                            <div class="recommendation-item">Aggressive medical therapy needed</div>
                            <div class="recommendation-item">Dual antiplatelet therapy (consult doctor)</div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="recommendation-item">Regular cardiology follow-up</div>
                            <div class="recommendation-item">Optimal medical therapy (statins, aspirin)</div>
                            <div class="recommendation-item">Lifestyle modification crucial</div>
                            <div class="recommendation-item">Repeat imaging in 1-2 years</div>
                            """, unsafe_allow_html=True)
                    
                    elif param_name == "Age":
                        if param_value >= 65:
                            st.markdown("""
                            <div class="recommendation-item">Annual cardiac risk assessment</div>
                            <div class="recommendation-item">Regular screening for silent ischemia</div>
                            <div class="recommendation-item">Medication review for age-appropriate dosing</div>
                            <div class="recommendation-item">Fall prevention and balance training</div>
                            <div class="recommendation-item">Regular health maintenance visits</div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="recommendation-item">Focus on primary prevention</div>
                            <div class="recommendation-item">Regular cardiovascular screening</div>
                            <div class="recommendation-item">Healthy lifestyle establishment</div>
                            <div class="recommendation-item">Risk factor modification</div>
                            """, unsafe_allow_html=True)
                    
                    elif param_name == "Exercise angina":
                        st.markdown("""
                        <div class="recommendation-item">**Cardiology evaluation within 2 weeks**</div>
                        <div class="recommendation-item">Stress testing with imaging</div>
                        <div class="recommendation-item">Sublingual nitroglycerin prescription</div>
                        <div class="recommendation-item">Avoid activities that trigger pain</div>
                        <div class="recommendation-item">Cardiac rehab program referral</div>
                        """, unsafe_allow_html=True)
                
                # General recommendations based on overall risk
                st.markdown('<div class="recommendation-category">General Recommendations</div>', unsafe_allow_html=True)
                
                if is_high_risk:
                    st.markdown("""
                    <div class="recommendation-item">Schedule cardiology appointment within 1-2 weeks</div>
                    <div class="recommendation-item">Carry nitroglycerin if prescribed</div>
                    <div class="recommendation-item">Learn CPR and have family trained</div>
                    <div class="recommendation-item">Create emergency action plan</div>
                    <div class="recommendation-item">Regular follow-up every 3-6 months</div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="recommendation-item">Annual physical with primary care physician</div>
                    <div class="recommendation-item">Regular BP monitoring (weekly)</div>
                    <div class="recommendation-item">Heart-healthy diet (Mediterranean style)</div>
                    <div class="recommendation-item">150 minutes moderate exercise weekly</div>
                    <div class="recommendation-item">Stress management techniques</div>
                    """, unsafe_allow_html=True)
            
            else:
                # No high risk parameters
                st.write("**All parameters are within normal ranges. Maintain heart-healthy habits:**")
                
                st.markdown("""
                <div class="recommendation-item">Continue regular exercise (150 min/week)</div>
                <div class="recommendation-item">Maintain healthy diet (fruits, vegetables, whole grains)</div>
                <div class="recommendation-item">Annual check-ups with primary care</div>
                <div class="recommendation-item">Monitor blood pressure regularly</div>
                <div class="recommendation-item">Avoid tobacco and limit alcohol</div>
                <div class="recommendation-item">Manage stress through relaxation techniques</div>
                <div class="recommendation-item">Maintain healthy weight (BMI 18.5-24.9)</div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # --- MODEL PERFORMANCE INFO ---
            st.markdown("---")
            with st.expander("📊 Model Performance Details"):
                col_info1, col_info2, col_info3 = st.columns(3)
                
                with col_info1:
                    st.metric("Accuracy", "94.6%")
                    st.metric("Precision", "88.4%")
                
                with col_info2:
                    st.metric("Recall", "90.6%")
                    st.metric("F1 Score", "89.4%")
                
                with col_info3:
                    st.metric("ROC-AUC", "0.982")
                    st.metric("Specificity", "95.9%")
                
                st.caption("Based on test dataset with 2,060 samples")
            
            # --- DISCLAIMER ---
            st.markdown("---")
            st.warning("""
            **⚠️ IMPORTANT MEDICAL DISCLAIMER:**
            
            This tool provides **risk estimation only** and is NOT a medical diagnosis. 
            
            **EMERGENCY SYMPTOMS (Call 911 or go to ER):**
            - Chest pain, pressure, or discomfort
            - Pain radiating to arm, neck, jaw, or back
            - Sudden shortness of breath
            - Fainting or near-fainting
            - Severe palpitations
            - Unexplained sweating with chest discomfort
            
            Always consult with qualified healthcare professionals for medical diagnosis and treatment.
            """)
            
        except PermissionError as e:
            st.error("""
            ❌ **Authentication Failed**
            
            Unable to access the prediction model. Please contact the system administrator.
            """)
            
        except FileNotFoundError as e:
            st.error("""
            ❌ **Model Not Available**
            
            Prediction model resources are currently unavailable.
            """)
            
        except Exception as e:
            st.error(f"❌ Error: Unable to complete prediction. Please try again later.")
            st.code(traceback.format_exc())

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>❤️ <b>Heart Disease Prediction System</b> | XGBoost Model v1.0</p>
    <p><small>For educational and decision-support purposes only | Not for clinical diagnosis</small></p>
</div>
""", unsafe_allow_html=True)