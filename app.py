import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="AI Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS Styling
# ----------------------------
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 2rem 0;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        color: #000000;
    }
    .gradient-title {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 800;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Header Section
# ----------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align: center; color: white; font-size: 3rem; font-weight: 800;'>🚗 AI Car Price Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #aaa;'>Powered by Deep Learning Neural Networks</p>", unsafe_allow_html=True)

st.markdown("---")

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_prediction_model():
    try:
        model = load_model("car_price_model_tf.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_prediction_model()

# ----------------------------
# Initialize Scalers
# ----------------------------
@st.cache_resource
def initialize_scalers():
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_X.fit([[18, 0, 0, 0, 0], [100, 200000, 50000, 1000000, 50000]])
    scaler_y.fit([[0], [100000]])
    return scaler_X, scaler_y

scaler_X, scaler_y = initialize_scalers()

# ----------------------------
# Sidebar - Information & Settings
# ----------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/car.png", width=80)
    st.title("About This App")
    st.markdown("""
    This advanced AI-powered application predicts car prices based on customer financial profiles using a deep learning neural network.
    
    **Features:**
    - 🧠 Deep Learning Model
    - 📊 Interactive Visualizations
    - 💡 Smart Recommendations
    - 📈 Detailed Analytics
    """)
    
    st.markdown("---")
    st.subheader("Model Information")
    st.info("""
    **Architecture:** TensorFlow/Keras Neural Network
    **Input Features:** 5 financial indicators
    **Training Data:** Historical car purchase records
    """)
    
    st.markdown("---")
    st.subheader("How It Works")
    st.markdown("""
    1. Enter customer financial details
    2. AI analyzes the data patterns
    3. Get instant price prediction
    4. View detailed insights & charts
    """)

# ----------------------------
# Main Content - Two Column Layout
# ----------------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📝 Customer Financial Profile")
    
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        age = st.slider("👤 Customer Age", min_value=18, max_value=100, value=30, help="Age of the customer in years")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        annual_salary = st.number_input(
            "💰 Annual Salary ($)",
            min_value=0,
            max_value=500000,
            value=50000,
            step=5000,
            help="Customer's yearly gross income"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        credit_card_debt = st.number_input(
            "💳 Credit Card Debt ($)",
            min_value=0,
            max_value=100000,
            value=5000,
            step=500,
            help="Outstanding credit card balance"
        )
        st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.subheader("💼 Additional Details")
    
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        net_worth = st.number_input(
            "🏦 Net Worth ($)",
            min_value=0,
            max_value=5000000,
            value=100000,
            step=10000,
            help="Total assets minus liabilities"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        current_car_value = st.number_input(
            "🚙 Current Car Value ($)",
            min_value=0,
            max_value=200000,
            value=15000,
            step=1000,
            help="Current market value of customer's car"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("**💡 Tip:** All values are in USD. Ensure accurate data entry for best prediction results.")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Financial Ratios Display
# ----------------------------
st.markdown("---")
st.subheader("📊 Financial Health Indicators")

col1, col2, col3, col4 = st.columns(4)

debt_to_income = (credit_card_debt / annual_salary * 100) if annual_salary > 0 else 0
car_to_salary = (current_car_value / annual_salary * 100) if annual_salary > 0 else 0
savings_capacity = annual_salary - credit_card_debt if annual_salary > credit_card_debt else 0
affordability_score = min(100, (net_worth / 10000) + (annual_salary / 1000))

with col1:
    st.metric("Debt-to-Income Ratio", f"{debt_to_income:.1f}%", 
              delta="Good" if debt_to_income < 30 else "High",
              delta_color="normal" if debt_to_income < 30 else "inverse")

with col2:
    st.metric("Car Value Ratio", f"{car_to_salary:.1f}%",
              delta="Reasonable" if car_to_salary < 50 else "High",
              delta_color="normal" if car_to_salary < 50 else "inverse")

with col3:
    st.metric("Net Savings Capacity", f"${savings_capacity:,.0f}",
              delta="Positive" if savings_capacity > 0 else "Negative",
              delta_color="normal" if savings_capacity > 0 else "inverse")

with col4:
    st.metric("Affordability Score", f"{affordability_score:.0f}/100",
              delta="Strong" if affordability_score > 60 else "Moderate",
              delta_color="normal" if affordability_score > 60 else "inverse")

# ----------------------------
# Prediction Button & Results
# ----------------------------
st.markdown("---")

if st.button("🎯 Predict Car Price", use_container_width=True):
    if model is None:
        st.error("Model not loaded. Please check the model file.")
    else:
        with st.spinner("🔮 AI is analyzing the data..."):
            # Prepare input
            input_data = pd.DataFrame(
                [[age, annual_salary, credit_card_debt, net_worth, current_car_value]],
                columns=['age', 'annual_Salary', 'credit_card_debt', 'net_worth', 'current_car_value']
            )
            
            # Scale and predict
            input_scaled = scaler_X.transform(input_data)
            pred_scaled = model.predict(input_scaled, verbose=0)
            predicted_price = scaler_y.inverse_transform(pred_scaled)[0][0]
            
            # Display main prediction
            st.markdown(f"""
                <div class="prediction-box">
                    <h2 style="margin: 0; color: white;">Predicted Car Price</h2>
                    <h1 style="margin: 10px 0; font-size: 3.5rem; color: white;">${predicted_price:,.2f}</h1>
                    <p style="margin: 0; font-size: 1.1rem;">Based on AI analysis of financial profile</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Price range estimation
            lower_bound = predicted_price * 0.90
            upper_bound = predicted_price * 1.10
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Conservative Estimate", f"${lower_bound:,.2f}", delta="-10%")
            with col2:
                st.metric("Predicted Price", f"${predicted_price:,.2f}", delta="Base")
            with col3:
                st.metric("Optimistic Estimate", f"${upper_bound:,.2f}", delta="+10%")
            
            # ----------------------------
            # Visualizations
            # ----------------------------
            st.markdown("---")
            st.subheader("📈 Detailed Analysis & Insights")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Price Range Chart
                fig_range = go.Figure()
                fig_range.add_trace(go.Bar(
                    x=['Conservative', 'Predicted', 'Optimistic'],
                    y=[lower_bound, predicted_price, upper_bound],
                    marker_color=['#FFA07A', '#667eea', '#90EE90'],
                    text=[f'${lower_bound:,.0f}', f'${predicted_price:,.0f}', f'${upper_bound:,.0f}'],
                    textposition='auto',
                ))
                fig_range.update_layout(
                    title="Price Range Estimation",
                    xaxis_title="Estimate Type",
                    yaxis_title="Price ($)",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_range, use_container_width=True)
            
            with viz_col2:
                # Financial Profile Radar
                categories = ['Age Factor', 'Income', 'Debt Level', 'Net Worth', 'Current Asset']
                values = [
                    (age / 100) * 100,
                    (annual_salary / 200000) * 100,
                    100 - (credit_card_debt / 50000) * 100,
                    (net_worth / 1000000) * 100,
                    (current_car_value / 50000) * 100
                ]
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    marker_color='#667eea'
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    title="Financial Profile Breakdown",
                    height=400
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            
            # Input vs Output Comparison
            st.markdown("---")
            st.subheader("🔍 Input Data Summary")
            
            summary_data = {
                'Financial Metric': ['Age', 'Annual Salary', 'Credit Card Debt', 'Net Worth', 'Current Car Value'],
                'Value': [f"{age} years", f"${annual_salary:,}", f"${credit_card_debt:,}", 
                         f"${net_worth:,}", f"${current_car_value:,}"],
                'Impact on Price': ['Moderate', 'High', 'Negative', 'High', 'Moderate']
            }
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True, hide_index=True)
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 Smart Recommendations")
            
            recommendations = []
            
            if debt_to_income > 30:
                recommendations.append("⚠️ **High Debt Ratio**: Consider reducing credit card debt before major purchase.")
            
            if predicted_price > annual_salary * 0.5:
                recommendations.append("💰 **Price Alert**: Predicted price exceeds 50% of annual salary. Budget carefully.")
            
            if net_worth < predicted_price * 0.3:
                recommendations.append("🏦 **Net Worth Check**: Consider building more savings before purchase.")
            
            if current_car_value < 5000 and predicted_price > 30000:
                recommendations.append("📊 **Upgrade Alert**: Significant upgrade from current vehicle. Ensure financial readiness.")
            
            if affordability_score > 70:
                recommendations.append("✅ **Strong Profile**: Financial indicators suggest good affordability for predicted price range.")
            
            if not recommendations:
                recommendations.append("✅ **Balanced Profile**: Financial metrics look healthy for this purchase range.")
            
            for rec in recommendations:
                st.markdown(f'<div class="info-box">{rec}</div>', unsafe_allow_html=True)
            
            # Download Report
            st.markdown("---")
            report_data = {
                'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                'Age': [age],
                'Annual Salary': [annual_salary],
                'Credit Card Debt': [credit_card_debt],
                'Net Worth': [net_worth],
                'Current Car Value': [current_car_value],
                'Predicted Price': [predicted_price],
                'Lower Bound': [lower_bound],
                'Upper Bound': [upper_bound],
                'Affordability Score': [affordability_score]
            }
            df_report = pd.DataFrame(report_data)
            
            csv = df_report.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Prediction Report (CSV)",
                data=csv,
                file_name=f"car_price_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>AI Car Price Predictor v2.0</strong></p>
        <p>Powered by TensorFlow & Deep Learning | Built with Streamlit</p>
        <p style='font-size: 0.9rem;'>⚠️ Predictions are estimates based on AI analysis. Always consult with financial advisors for major purchases.</p>
    </div>
""", unsafe_allow_html=True)