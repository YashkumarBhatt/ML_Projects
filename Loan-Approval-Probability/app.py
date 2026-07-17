# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# ==========================================
# 1. PAGE SETUP & STYLING
# ==========================================
st.set_page_config(
    page_title="Apex Loan Systems | Analytics & Eligibility Engine", 
    page_icon="🏦", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom premium CSS injection
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global Styling */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        color: #334155;
    }
    
    /* Header styling */
    .title-container {
        padding: 2rem;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(240, 249, 255, 0.95) 100%);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 0, 0, 0.05);
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
    }
    .gradient-title {
        background: linear-gradient(90deg, #4f46e5 0%, #9333ea 50%, #db2777 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
    }
    .subtitle {
        color: #475569;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Glassmorphic card styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(0, 0, 0, 0.05);
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.05);
    }
    
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1e293b;
        border-left: 4px solid #4f46e5;
        padding-left: 10px;
        margin-bottom: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Metrics display */
    .live-metric-box {
        background: rgba(79, 70, 229, 0.05);
        border: 1px solid rgba(79, 70, 229, 0.2);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .live-metric-box:hover {
        transform: translateY(-2px);
        border-color: rgba(79, 70, 229, 0.4);
    }
    .live-metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4f46e5;
    }
    .live-metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.2rem;
    }
    
    /* Prediction displays */
    .result-card {
        border-radius: 16px;
        padding: 2rem;
        margin-top: 1rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
    }
    .result-approved {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .result-rejected {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    
    .status-badge {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .status-badge-approved {
        color: #059669;
    }
    .status-badge-rejected {
        color: #dc2626;
    }
    
    /* Progress/Meter styling */
    .meter-container {
        background-color: rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        height: 12px;
        width: 100%;
        margin: 0.8rem 0;
        overflow: hidden;
    }
    .meter-bar-approved {
        background: linear-gradient(90deg, #10b981, #34d399);
        height: 100%;
        border-radius: 8px;
    }
    .meter-bar-rejected {
        background: linear-gradient(90deg, #ef4444, #f87171);
        height: 100%;
        border-radius: 8px;
    }
    
    /* Custom spacing & clean dividers */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, rgba(0,0,0,0) 0%, rgba(0,0,0,0.1) 50%, rgba(0,0,0,0) 100%);
        margin: 2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 2. DATA & MODEL CACHING
# ==========================================
@st.cache_resource
def load_ml_pipeline():
    try:
        base_path = os.path.dirname(__file__)
        preprocessor = joblib.load(os.path.join(base_path, 'loan_preprocessor.joblib'))
        model = joblib.load(os.path.join(base_path, 'loan_tree_model.joblib'))
        return preprocessor, model
    except Exception as e:
        st.error(f"Failed to load machine learning artifacts: {e}")
        return None, None

@st.cache_data
def load_historical_data():
    try:
        base_path = os.path.dirname(__file__)
        df = pd.read_csv(os.path.join(base_path, 'loan_approval.csv'))
        return df
    except Exception as e:
        st.error(f"Failed to load historical data: {e}")
        return None

preprocessor, model = load_ml_pipeline()
historical_data = load_historical_data()

# ==========================================
# 3. APP BANNER
# ==========================================
st.markdown(
    """
    <div class="title-container">
        <h1 class="gradient-title">APEX CREDIT SYSTEMS</h1>
        <p class="subtitle">🏦 Intelligent Underwriting & Portfolio Decision Engine</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Tab Selection
tab_eligibility, tab_dashboard, tab_calculator = st.tabs([
    "🏦 Eligibility Engine", 
    "📊 Portfolio Dashboard", 
    "🧮 Amortization Planner"
])

# ==========================================
# TAB 1: ELIGIBILITY ENGINE
# ==========================================
with tab_eligibility:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Applicant Underwriting Intake</div>', unsafe_allow_html=True)
    
    # Grid columns
    col_pers, col_fin, col_req = st.columns(3)
    
    with col_pers:
        st.write("### 👤 Personal Profile")
        education = st.selectbox(
            "Education Status", 
            ["Graduate", "Not Graduate"],
            help="Select the applicant's highest completed level of education."
        )
        marital_status = st.selectbox(
            "Marital Status", 
            ["Married", "Single"],
            help="Current legal marital status of the applicant."
        )
        dependents = st.selectbox(
            "Number of Dependents", 
            ["0", "1", "2", "3+"],
            help="Number of financially dependent family members."
        )
        property_area = st.selectbox(
            "Property Location", 
            ["Urban", "Semiurban", "Rural"],
            help="Location classification of the property being financed."
        )

    with col_fin:
        st.write("### 💼 Financial Standing")
        employment_type = st.selectbox(
            "Employment Class", 
            ["Salaried", "Self Employed"],
            help="Primary income structure."
        )
        applicant_income = st.number_input(
            "Applicant Annual Income ($)", 
            min_value=0, 
            max_value=None,
            value=65000, 
            step=5000,
            help="Gross annual income of the primary applicant."
        )
        coapplicant_income = st.number_input(
            "Co-applicant Annual Income ($)", 
            min_value=0, 
            max_value=None,
            value=0, 
            step=5000,
            help="Gross annual income of the co-applicant (if applicable)."
        )

    with col_req:
        st.write("### 📄 Loan Request")
        loan_amount = st.number_input(
            "Loan Amount Requested ($)", 
            min_value=5000, 
            max_value=None,
            value=180000, 
            step=10000,
            help="Total principal requested."
        )
        loan_amount_term = st.selectbox(
            "Loan Term Duration", 
            [12, 36, 60, 84, 120, 180, 240, 300, 360, 480], 
            index=8, # Default 360 months (30 years)
            help="Duration of the loan in months."
        )
        credit_history_mapped = st.selectbox(
            "Credit History standing", 
            ["Excellent (No Defaults / Good History)", "Poor / No Established Credit History"],
            help="Historical credit performance status."
        )
        credit_history = 1.0 if "Excellent" in credit_history_mapped else 0.0

    st.markdown('</div>', unsafe_allow_html=True)
    
    # Real-time Metrics and Estimations
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Pre-Underwriting Indicators (Live Estimates)</div>', unsafe_allow_html=True)
    
    total_income = applicant_income + coapplicant_income
    annual_mortgage_factor = 0.065 # assumed interest rate of 6.5% for general estimation
    
    # Calculate estimated monthly mortgage payment (Principal + Interest)
    r = annual_mortgage_factor / 12
    n = loan_amount_term
    if r > 0:
        est_monthly_payment = loan_amount * (r * (1 + r)**n) / ((1 + r)**n - 1)
    else:
        est_monthly_payment = loan_amount / n
        
    monthly_combined_income = total_income / 12
    dti_ratio = (est_monthly_payment / monthly_combined_income * 100) if monthly_combined_income > 0 else 0
    lti_ratio = (loan_amount / total_income) if total_income > 0 else 0
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.markdown(
            f"""
            <div class="live-metric-box">
                <div class="live-metric-value">${total_income:,.0f}</div>
                <div class="live-metric-label">Combined Annual Income</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with metric_col2:
        st.markdown(
            f"""
            <div class="live-metric-box">
                <div class="live-metric-value">${est_monthly_payment:,.2f}</div>
                <div class="live-metric-label">Est. Monthly Payment</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with metric_col3:
        dti_color = "#34d399" if dti_ratio <= 36 else "#fbbf24" if dti_ratio <= 43 else "#f87171"
        st.markdown(
            f"""
            <div class="live-metric-box">
                <div class="live-metric-value" style="color: {dti_color};">{dti_ratio:.1f}%</div>
                <div class="live-metric-label">Estimated Debt-to-Income (DTI)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with metric_col4:
        lti_color = "#34d399" if lti_ratio <= 3.0 else "#fbbf24" if lti_ratio <= 4.5 else "#f87171"
        st.markdown(
            f"""
            <div class="live-metric-box">
                <div class="live-metric-value" style="color: {lti_color};">{lti_ratio:.2f}x</div>
                <div class="live-metric-label">Loan-to-Income Ratio (LTI)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Predict button centered
    st.markdown('<div style="text-align: center; margin-top: 2rem; margin-bottom: 2rem;">', unsafe_allow_html=True)
    trigger_predict = st.button("Evaluate Decision Pipeline", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Evaluation execution
    if trigger_predict:
        if preprocessor is None or model is None:
            st.error("Engine failure: Models are not loaded.")
        else:
            # Package fields matching preprocessing expected schema
            input_data = {
                'ApplicantIncome': applicant_income,
                'CoapplicantIncome': coapplicant_income,
                'LoanAmount': loan_amount,
                'Loan_Amount_Term': loan_amount_term,
                'CreditHistory': credit_history,
                'Education': education,
                'EmploymentType': employment_type,
                'MaritalStatus': marital_status,
                'Dependents': dependents,
                'PropertyArea': property_area
            }
            
            input_df = pd.DataFrame([input_data])
            
            try:
                # Transform using standard pipeline preprocessor
                transformed_data = preprocessor.transform(input_df)
                
                # Predict
                prediction = model.predict(transformed_data)[0]
                probabilities = model.predict_proba(transformed_data)[0]
                
                if prediction == 1:
                    conf = probabilities[1] * 100
                    st.markdown(
                        f"""
                        <div class="result-card result-approved">
                            <div class="status-badge status-badge-approved">✅ APPLICATION APPROVED</div>
                            <p style="font-size: 1.1rem; color: #000000; margin-bottom: 1.5rem;">
                                The algorithmic underwriting model has approved this profile with a <strong>{conf:.1f}%</strong> confidence rating.
                            </p>
                            <div class="meter-container">
                                <div class="meter-bar-approved" style="width: {conf}%;"></div>
                            </div>
                            <div style="font-size: 0.9rem; color: #94a3b8; text-align: right; margin-top: 0.2rem;">
                                Confidence Score: {conf:.1f}%
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    st.info("💡 **Underwriting Notes:** Applicant possesses an excellent combination of historical credit standing, manageable loan-to-income ratio, and stable demographic variables. Application is routed to automatic final processing.")
                    
                else:
                    conf = probabilities[0] * 100
                    st.markdown(
                        f"""
                        <div class="result-card result-rejected">
                            <div class="status-badge status-badge-rejected">⚠️ APPLICATION REJECTED</div>
                            <p style="font-size: 1.1rem; color: #000000; margin-bottom: 1.5rem;">
                                The algorithmic underwriting model has rejected this profile with a <strong>{conf:.1f}%</strong> rejection probability.
                            </p>
                            <div class="meter-container">
                                <div class="meter-bar-rejected" style="width: {conf}%;"></div>
                            </div>
                            <div style="font-size: 0.9rem; color: #94a3b8; text-align: right; margin-top: 0.2rem;">
                                Model Decision Confidence: {conf:.1f}%
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("### 🔍 Risk Mitigation Recommendations")
                    recomm = []
                    
                    if credit_history == 0.0:
                        recomm.append("🔴 **Establish/Improve Credit standing:** Credit History is the most significant indicator. Consider cleaning outstanding debts or obtaining secured lines of credit before re-evaluating.")
                    if dti_ratio > 43:
                        recomm.append("🟡 **Lower requested Loan Amount:** The current monthly payments constitute a high portion of combined income. Reducing the loan request by **15-20%** will bring DTI back into standard risk guidelines.")
                    if lti_ratio > 4.5:
                        recomm.append("🟡 **Add a Qualified Co-Applicant:** Adding an extra borrower with solid income streams or high credit standing will dramatically improve the approval probabilities.")
                    if loan_amount_term < 180:
                        recomm.append("🟢 **Extend Loan Duration:** Extending the repayment term lowers your periodic payment burden and reduces DTI risk.")
                        
                    if not recomm:
                        recomm.append("⚪ **Generic Optimization:** Try slightly lowering the loan amount requested, or choosing a different property target to optimize profile factors.")
                        
                    for r_item in recomm:
                        st.write(r_item)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Decision engine crashed: {e}")

# ==========================================
# TAB 2: PORTFOLIO DASHBOARD
# ==========================================
with tab_dashboard:
    if historical_data is None:
        st.warning("Historical data is unavailable.")
    else:
        # KPI widgets
        total_records = len(historical_data)
        approval_rate = (historical_data['Loan_Status'].mean()) * 100
        avg_income = historical_data['ApplicantIncome'].mean()
        avg_loan = historical_data['LoanAmount'].mean()
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Portfolio Indicators (Historical Database Analysis)</div>', unsafe_allow_html=True)
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            st.markdown(
                f"""
                <div class="live-metric-box">
                    <div class="live-metric-value">{total_records:,}</div>
                    <div class="live-metric-label">Applications Checked</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with kpi_col2:
            st.markdown(
                f"""
                <div class="live-metric-box">
                    <div class="live-metric-value">{approval_rate:.1f}%</div>
                    <div class="live-metric-label">Base Approval Rate</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with kpi_col3:
            st.markdown(
                f"""
                <div class="live-metric-box">
                    <div class="live-metric-value">${avg_income:,.0f}</div>
                    <div class="live-metric-label">Average Ann. Income</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with kpi_col4:
            st.markdown(
                f"""
                <div class="live-metric-box">
                    <div class="live-metric-value">${avg_loan:,.0f}</div>
                    <div class="live-metric-label">Average Loan Requested</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Grid of Charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 💳 The Critical Role of Credit History")
            
            # Group by credit history and calculate approval rate
            cred_grouped = historical_data.groupby('CreditHistory')['Loan_Status'].mean().reset_index()
            cred_grouped['CreditHistory'] = cred_grouped['CreditHistory'].map({1: 'Excellent History (1.0)', 0: 'Poor/No History (0.0)'})
            cred_grouped['Approval Rate (%)'] = cred_grouped['Loan_Status'] * 100
            
            fig1 = px.bar(
                cred_grouped,
                x='CreditHistory',
                y='Approval Rate (%)',
                color='CreditHistory',
                color_discrete_map={'Excellent History (1.0)': '#10b981', 'Poor/No History (0.0)': '#ef4444'},
                text='Approval Rate (%)',
                labels={'CreditHistory': 'Credit History Status'}
            )
            fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig1.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_family='Outfit',
                font_color='#f8fafc',
                showlegend=False,
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', range=[0, 110])
            )
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with chart_col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### ⚖️ Income vs. Loan Amount Decision Clusters")
            
            sample_df = historical_data.sample(min(800, len(historical_data)), random_state=42)
            sample_df['Approved'] = sample_df['Loan_Status'].map({1: 'Approved', 0: 'Rejected'})
            
            fig2 = px.scatter(
                sample_df,
                x='ApplicantIncome',
                y='LoanAmount',
                color='Approved',
                color_discrete_map={'Approved': '#34d399', 'Rejected': '#f87171'},
                opacity=0.6,
                labels={'ApplicantIncome': 'Applicant Annual Income ($)', 'LoanAmount': 'Loan Amount ($)'}
            )
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_family='Outfit',
                font_color='#f8fafc',
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🎓 Demographic Acceptance Matrices")
        
        # Cross tab of Property Area & Education with Approval Rate
        cross_tab = historical_data.groupby(['PropertyArea', 'Education'])['Loan_Status'].mean().reset_index()
        cross_tab['Approval Rate (%)'] = cross_tab['Loan_Status'] * 100
        
        fig3 = px.bar(
            cross_tab,
            x='PropertyArea',
            y='Approval Rate (%)',
            color='Education',
            barmode='group',
            color_discrete_sequence=['#6366f1', '#ec4899'],
            labels={'PropertyArea': 'Property Area Classification'}
        )
        fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_family='Outfit',
            font_color='#f8fafc',
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', range=[0, 110]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 3: AMORTIZATION PLANNER
# ==========================================
with tab_calculator:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Financial Repayment Simulator</div>', unsafe_allow_html=True)
    
    col_calc1, col_calc2 = st.columns([1, 2])
    
    with col_calc1:
        st.write("### ⚙️ Simulator Inputs")
        # Pull request values as defaults for seamless experience
        c_amount = st.number_input(
            "Loan Principal Amount ($)", 
            min_value=5000, 
            max_value=None, 
            value=int(loan_amount), 
            step=5000
        )
        c_term = st.selectbox(
            "Repayment Term (Months)", 
            [12, 36, 60, 84, 120, 180, 240, 300, 360, 480], 
            index=[12, 36, 60, 84, 120, 180, 240, 300, 360, 480].index(loan_amount_term) if loan_amount_term in [12, 36, 60, 84, 120, 180, 240, 300, 360, 480] else 8
        )
        c_rate = st.slider(
            "Annual Interest Rate (APR %)", 
            min_value=1.0, 
            max_value=18.0, 
            value=6.5, 
            step=0.05
        )
        
        # Calculate payment metrics
        i_monthly = (c_rate / 100) / 12
        n_pmt = c_term
        if i_monthly > 0:
            monthly_pmt = c_amount * (i_monthly * (1 + i_monthly)**n_pmt) / ((1 + i_monthly)**n_pmt - 1)
        else:
            monthly_pmt = c_amount / n_pmt
            
        total_payout = monthly_pmt * n_pmt
        total_interest = total_payout - c_amount
        
        st.write("---")
        st.markdown(
            f"""
            <div style="margin-bottom: 0.8rem;">
                <span style="color: #94a3b8; font-size: 0.9rem;">Estimated Monthly Installment</span><br/>
                <span style="font-size: 1.6rem; font-weight: 700; color: #818cf8;">${monthly_pmt:,.2f}</span>
            </div>
            <div style="margin-bottom: 0.8rem;">
                <span style="color: #94a3b8; font-size: 0.9rem;">Total Principal Paid</span><br/>
                <span style="font-size: 1.2rem; font-weight: 600; color: #f8fafc;">${c_amount:,.2f}</span>
            </div>
            <div style="margin-bottom: 0.8rem;">
                <span style="color: #94a3b8; font-size: 0.9rem;">Total Interest Incurred</span><br/>
                <span style="font-size: 1.2rem; font-weight: 600; color: #f87171;">${total_interest:,.2f}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_calc2:
        st.write("### 📈 Amortization Projection")
        
        # Generate Amortization Schedule
        schedule_data = []
        remaining_balance = c_amount
        cumulative_interest = 0
        cumulative_principal = 0
        
        for m in range(1, n_pmt + 1):
            interest_payment = remaining_balance * i_monthly
            principal_payment = monthly_pmt - interest_payment
            
            # Correct edge cases on final payment
            if m == n_pmt:
                principal_payment = remaining_balance
                monthly_payment_actual = principal_payment + interest_payment
            else:
                monthly_payment_actual = monthly_pmt
                
            remaining_balance -= principal_payment
            cumulative_interest += interest_payment
            cumulative_principal += principal_payment
            
            schedule_data.append({
                'Month': m,
                'Payment': monthly_payment_actual,
                'Principal': principal_payment,
                'Interest': interest_payment,
                'Remaining Balance': max(0.0, remaining_balance),
                'Cumulative Interest': cumulative_interest,
                'Cumulative Principal': cumulative_principal
            })
            
        sched_df = pd.DataFrame(schedule_data)
        
        # Pie chart: Principal vs Interest
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Principal', 'Interest'],
            values=[c_amount, total_interest],
            hole=.4,
            marker_colors=['#6366f1', '#ef4444']
        )])
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_family='Outfit',
            font_color='#f8fafc',
            height=200,
            margin=dict(t=0, b=0, l=0, r=0),
            legend=dict(orientation="h", y=0)
        )
        
        # Line/Area Chart: Balances over time
        fig_balance = go.Figure()
        fig_balance.add_trace(go.Scatter(
            x=sched_df['Month'],
            y=sched_df['Remaining Balance'],
            fill='tozeroy',
            name='Remaining Principal',
            line_color='#6366f1'
        ))
        fig_balance.add_trace(go.Scatter(
            x=sched_df['Month'],
            y=sched_df['Cumulative Interest'],
            name='Cumulative Interest Paid',
            line_color='#ef4444'
        ))
        fig_balance.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_family='Outfit',
            font_color='#f8fafc',
            height=260,
            margin=dict(t=20, b=0, l=0, r=0),
            xaxis=dict(title='Month', showgrid=False),
            yaxis=dict(title='Balance ($)', gridcolor='rgba(255,255,255,0.05)'),
            legend=dict(orientation="h", y=1.1, x=0)
        )
        
        st.plotly_chart(fig_balance, use_container_width=True)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("📄 View Amortization Repayment Ledger"):
        display_sched_df = sched_df[['Month', 'Payment', 'Principal', 'Interest', 'Remaining Balance', 'Cumulative Interest']].copy()
        for col in ['Payment', 'Principal', 'Interest', 'Remaining Balance', 'Cumulative Interest']:
            display_sched_df[col] = display_sched_df[col].map('${:,.2f}'.format)
        st.dataframe(display_sched_df, use_container_width=True, hide_index=True)
