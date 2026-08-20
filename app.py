import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD MODEL, SCALERS AND FEATURES
# ============================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load("churn_logistic_model.pkl")

    # First scaler
    pre_scaler = joblib.load("churn_pre_scaler.pkl")

    # Final/model scaler
    scaler = joblib.load("churn_scaler.pkl")

    # Training feature columns
    feature_columns = joblib.load("churn_features.pkl")

    return model, pre_scaler, scaler, feature_columns


model, pre_scaler, scaler, feature_columns = load_artifacts()


# ============================================================
# NUMERICAL FEATURES
# ============================================================

numerical_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .footer {
        text-align: center;
        font-size: 14px;
        margin-top: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📊 Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Customer Churn Prediction System'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)


# ============================================================
# COLUMN 1
# ============================================================

with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12,
        step=1
    )


# ============================================================
# COLUMN 2
# ============================================================

with col2:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )


# ============================================================
# COLUMN 3
# ============================================================

with col3:

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


# ============================================================
# BILLING INFORMATION
# ============================================================

st.divider()

st.subheader("💳 Billing & Contract Information")

col4, col5, col6 = st.columns(3)


# ============================================================
# CONTRACT
# ============================================================

with col4:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )


# ============================================================
# PAYMENT METHOD
# ============================================================

with col5:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


# ============================================================
# CHARGES
# ============================================================

with col6:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0,
        step=10.0
    )


# ============================================================
# CREATE RAW INPUT
# ============================================================

def prepare_input():

    input_data = pd.DataFrame({

        "gender": [gender],

        "SeniorCitizen": [senior_citizen],

        "Partner": [partner],

        "Dependents": [dependents],

        "tenure": [tenure],

        "PhoneService": [phone_service],

        "MultipleLines": [multiple_lines],

        "InternetService": [internet_service],

        "OnlineSecurity": [online_security],

        "OnlineBackup": [online_backup],

        "DeviceProtection": [device_protection],

        "TechSupport": [tech_support],

        "StreamingTV": [streaming_tv],

        "StreamingMovies": [streaming_movies],

        "Contract": [contract],

        "PaperlessBilling": [paperless_billing],

        "PaymentMethod": [payment_method],

        "MonthlyCharges": [monthly_charges],

        "TotalCharges": [total_charges]
    })

    return input_data


# ============================================================
# ENCODING FUNCTION
# ============================================================

def encode_input(data):

    data = data.copy()


    # --------------------------------------------------------
    # Binary columns
    # --------------------------------------------------------

    data["gender_Male"] = (
        data["gender"] == "Male"
    ).astype(int)

    data["Partner_Yes"] = (
        data["Partner"] == "Yes"
    ).astype(int)

    data["Dependents_Yes"] = (
        data["Dependents"] == "Yes"
    ).astype(int)

    data["PhoneService_Yes"] = (
        data["PhoneService"] == "Yes"
    ).astype(int)

    data["PaperlessBilling_Yes"] = (
        data["PaperlessBilling"] == "Yes"
    ).astype(int)


    # --------------------------------------------------------
    # Multiple Lines
    # --------------------------------------------------------

    data["MultipleLines_No phone service"] = (
        data["MultipleLines"] == "No phone service"
    ).astype(int)

    data["MultipleLines_Yes"] = (
        data["MultipleLines"] == "Yes"
    ).astype(int)


    # --------------------------------------------------------
    # Internet Service
    # --------------------------------------------------------

    data["InternetService_Fiber optic"] = (
        data["InternetService"] == "Fiber optic"
    ).astype(int)

    data["InternetService_No"] = (
        data["InternetService"] == "No"
    ).astype(int)


    # --------------------------------------------------------
    # Online Security
    # --------------------------------------------------------

    data["OnlineSecurity_No internet service"] = (
        data["OnlineSecurity"] == "No internet service"
    ).astype(int)

    data["OnlineSecurity_Yes"] = (
        data["OnlineSecurity"] == "Yes"
    ).astype(int)


    # --------------------------------------------------------
    # Online Backup
    # --------------------------------------------------------

    data["OnlineBackup_No internet service"] = (
        data["OnlineBackup"] == "No internet service"
    ).astype(int)

    data["OnlineBackup_Yes"] = (
        data["OnlineBackup"] == "Yes"
    ).astype(int)


    # --------------------------------------------------------
    # Device Protection
    # --------------------------------------------------------

    data["DeviceProtection_No internet service"] = (
        data["DeviceProtection"] == "No internet service"
    ).astype(int)

    data["DeviceProtection_Yes"] = (
        data["DeviceProtection"] == "Yes"
    ).astype(int)


    # --------------------------------------------------------
    # Tech Support
    # --------------------------------------------------------

    data["TechSupport_No internet service"] = (
        data["TechSupport"] == "No internet service"
    ).astype(int)

    data["TechSupport_Yes"] = (
        data["TechSupport"] == "Yes"
    ).astype(int)


    # --------------------------------------------------------
    # Streaming TV
    # --------------------------------------------------------

    data["StreamingTV_No internet service"] = (
        data["StreamingTV"] == "No internet service"
    ).astype(int)

    data["StreamingTV_Yes"] = (
        data["StreamingTV"] == "Yes"
    ).astype(int)


    # --------------------------------------------------------
    # Streaming Movies
    # --------------------------------------------------------

    data["StreamingMovies_No internet service"] = (
        data["StreamingMovies"] == "No internet service"
    ).astype(int)

    data["StreamingMovies_Yes"] = (
        data["StreamingMovies"] == "Yes"
    ).astype(int)


    # --------------------------------------------------------
    # Contract
    # --------------------------------------------------------

    data["Contract_One year"] = (
        data["Contract"] == "One year"
    ).astype(int)

    data["Contract_Two year"] = (
        data["Contract"] == "Two year"
    ).astype(int)


    # --------------------------------------------------------
    # Payment Method
    # --------------------------------------------------------

    data["PaymentMethod_Credit card (automatic)"] = (
        data["PaymentMethod"] == "Credit card (automatic)"
    ).astype(int)

    data["PaymentMethod_Electronic check"] = (
        data["PaymentMethod"] == "Electronic check"
    ).astype(int)

    data["PaymentMethod_Mailed check"] = (
        data["PaymentMethod"] == "Mailed check"
    ).astype(int)


    # --------------------------------------------------------
    # Remove original categorical columns
    # --------------------------------------------------------

    categorical_columns = [

        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"
    ]

    data = data.drop(
        columns=categorical_columns
    )


    # --------------------------------------------------------
    # Exact training column order
    # --------------------------------------------------------

    data = data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return data


# ============================================================
# PREDICT BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Customer Churn",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # STEP 1: Create raw input
        # ----------------------------------------------------

        input_data = prepare_input()


        # ----------------------------------------------------
        # STEP 2: First scaling
        # ----------------------------------------------------

        input_data[numerical_features] = (
            pre_scaler.transform(
                input_data[numerical_features]
            )
        )


        # ----------------------------------------------------
        # STEP 3: Encoding
        # ----------------------------------------------------

        input_encoded = encode_input(
            input_data
        )


        # ----------------------------------------------------
        # STEP 4: Verify features
        # ----------------------------------------------------

        if list(input_encoded.columns) != list(feature_columns):

            st.error(
                "❌ Feature columns do not match training columns."
            )

            st.stop()


        # ----------------------------------------------------
        # STEP 5: Final scaling
        # ----------------------------------------------------

        input_scaled = scaler.transform(
            input_encoded
        )


        # ----------------------------------------------------
        # STEP 6: Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            input_scaled
        )[0]


        # ----------------------------------------------------
        # STEP 7: Probability
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            input_scaled
        )[0]

        no_churn_probability = probabilities[0]

        churn_probability = probabilities[1]


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        st.subheader("📈 Prediction Result")


        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        result_col1, result_col2 = st.columns(2)


        with result_col1:

            st.metric(
                "No Churn Probability",
                f"{no_churn_probability * 100:.2f}%"
            )


        with result_col2:

            st.metric(
                "Churn Probability",
                f"{churn_probability * 100:.2f}%"
            )


        # ----------------------------------------------------
        # Final Prediction
        # ----------------------------------------------------

        if prediction == 1:

            st.error(
                "⚠️ Customer is likely to Churn"
            )

            st.warning(
                "This customer may be at risk of leaving the service."
            )

        else:

            st.success(
                "✅ Customer is likely to Stay"
            )

            st.info(
                "This customer is currently predicted to remain with the service."
            )


        # ----------------------------------------------------
        # Prediction Details
        # ----------------------------------------------------

        st.subheader("🔍 Prediction Details")

        st.write(
            "**Prediction:**",
            "Churn" if prediction == 1 else "No Churn"
        )

        st.write(
            "**Churn Probability:**",
            f"{churn_probability * 100:.6f}%"
        )

        st.write(
            "**No Churn Probability:**",
            f"{no_churn_probability * 100:.6f}%"
        )


        # ----------------------------------------------------
        # Model Input
        # ----------------------------------------------------

        with st.expander("🔎 View Model Input"):

            st.write(
                "Encoded input sent to the model:"
            )

            st.dataframe(
                input_encoded
            )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        Built with Python, Scikit-learn & Streamlit
        <br>
        Customer Churn Prediction Project
    </div>
    """,
    unsafe_allow_html=True
)