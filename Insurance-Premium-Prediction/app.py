import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ── Page config ──
st.set_page_config(page_title="Insurance Premium Calculator", layout="centered")

# ── Custom CSS for a polished, customer-facing look ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1e3a5f, #2e86de);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .main-header p {
        color: #64748b;
        font-size: 1.05rem;
        margin-top: 0;
    }

    .premium-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2e86de 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(30, 58, 95, 0.3);
    }
    .premium-card .label {
        font-size: 0.95rem;
        font-weight: 500;
        opacity: 0.85;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    .premium-card .amount {
        font-size: 3rem;
        font-weight: 700;
        margin: 0.25rem 0;
    }
    .premium-card .range {
        font-size: 0.95rem;
        opacity: 0.8;
        margin-top: 0.5rem;
    }

    .summary-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .summary-card h3 {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
    }
    .summary-row {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px solid #f1f5f9;
    }
    .summary-row:last-child {
        border-bottom: none;
    }
    .summary-row .key {
        color: #64748b;
        font-size: 0.95rem;
    }
    .summary-row .value {
        color: #1e293b;
        font-weight: 600;
        font-size: 0.95rem;
    }

    .info-note {
        background: #eff6ff;
        border-left: 4px solid #2e86de;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        margin-top: 1rem;
        font-size: 0.85rem;
        color: #1e40af;
    }

    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #1e3a5f 0%, #2e86de 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-size: 1.05rem;
        font-weight: 600;
        cursor: pointer;
        transition: transform 0.15s, box-shadow 0.15s;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46, 134, 222, 0.35);
    }
</style>
""", unsafe_allow_html=True)


# ── Train model (cached) ──
@st.cache_data
def load_and_train():
    df = pd.read_csv("insurance.csv")

    df["gender"] = df["gender"].map({"male": 1, "female": 0})
    df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
    region_dummies = pd.get_dummies(df["region"], drop_first=True, dtype=int)
    df = pd.concat([df.drop("region", axis=1), region_dummies], axis=1)

    X = df.drop("charges", axis=1)
    y = df["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeRegressor(
        min_samples_leaf=10,
        max_depth=7,
        min_samples_split=2,
        criterion="absolute_error",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = round(mean_absolute_error(y_test, y_pred), 2)

    return model, X.columns.tolist(), mae


model, feature_cols, mae = load_and_train()

# ── Header ──
st.markdown(
    """
    <div class="main-header">
        <h1>🏥 Insurance Premium Calculator</h1>
        <p>Get an instant estimate of your annual medical insurance premium</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ── Input form ──
st.subheader("📋 Your Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    smoker = st.selectbox("Do you smoke?", ["No", "Yes"])

with col2:
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0, step=1)
    region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

st.markdown("")  # spacer

# ── Predict button ──
predict_clicked = st.button("💰 Calculate My Premium")

if predict_clicked:
    # Encode inputs
    gender_enc = 1 if gender == "Male" else 0
    smoker_enc = 1 if smoker == "Yes" else 0
    region_lower = region.lower()
    region_northwest = 1 if region_lower == "northwest" else 0
    region_southeast = 1 if region_lower == "southeast" else 0
    region_southwest = 1 if region_lower == "southwest" else 0

    input_data = pd.DataFrame(
        [[age, bmi, children, gender_enc, smoker_enc,
          region_northwest, region_southeast, region_southwest]],
        columns=feature_cols,
    )

    prediction = model.predict(input_data)[0]
    low = max(prediction - mae, 0)
    high = prediction + mae

    # ── Premium card ──
    st.markdown(
        f"""
        <div class="premium-card">
            <div class="label">Your Estimated Annual Premium</div>
            <div class="amount">${prediction:,.2f}</div>
            <div class="range">📊 Estimated range: ${low:,.2f} – ${high:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Summary card ──
    st.markdown(
        f"""
        <div class="summary-card">
            <h3>📄 Your Profile Summary</h3>
            <div class="summary-row"><span class="key">Age</span><span class="value">{age} years</span></div>
            <div class="summary-row"><span class="key">Gender</span><span class="value">{gender}</span></div>
            <div class="summary-row"><span class="key">BMI</span><span class="value">{bmi}</span></div>
            <div class="summary-row"><span class="key">Smoker</span><span class="value">{smoker}</span></div>
            <div class="summary-row"><span class="key">Dependents</span><span class="value">{children}</span></div>
            <div class="summary-row"><span class="key">Region</span><span class="value">{region}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Info note ──
    st.markdown(
        f"""
        <div class="info-note">
            ℹ️ This is an indicative estimate based on the details you provided.
            Your actual premium will be finalised after a one-on-one consultation,
            taking into account your medical history and other underwriting parameters.
        </div>
        """,
        unsafe_allow_html=True,
    )
