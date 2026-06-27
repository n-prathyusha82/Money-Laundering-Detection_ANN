import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Money Laundering Detector",
    page_icon="🏦",
    layout="centered",
)

# ── Load artefacts ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    model = tf.keras.models.load_model("model.keras")
    return preprocessor, model

try:
    preprocessor, model = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    load_error = str(e)

# ── Helper: frequency encoding maps (same values used in training) ──────────────
# These are approximate from-bank and to-bank frequency maps from the dataset.
# In production, persist these alongside preprocessor.pkl.
FROM_BANK_FREQ_DEFAULT = 0.005   # fallback for unseen bank IDs
TO_BANK_FREQ_DEFAULT   = 0.001

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🏦 Money Laundering Detection")
st.markdown(
    "Enter transaction details below to check whether the transaction is "
    "**suspicious (potential laundering)** or **legitimate**."
)

if not artifacts_loaded:
    st.error(
        f"⚠️ Could not load model artefacts. "
        f"Make sure `preprocessor.pkl` and `model.keras` are in the same "
        f"directory as this script.\n\n`{load_error}`"
    )
    st.stop()

st.divider()

# ── Input form ─────────────────────────────────────────────────────────────────
with st.form("transaction_form"):
    st.subheader("Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        amount_received = st.number_input(
            "Amount Received", min_value=0.0, value=1000.0, step=0.01,
            help="Amount received by the destination account"
        )
        receiving_currency = st.selectbox(
            "Receiving Currency",
            ["US Dollar", "Euro", "Bitcoin", "Rupee", "Mexican Peso",
             "UK Pound", "Yen", "Ruble", "Yuan", "Australian Dollar",
             "Canadian Dollar", "Swiss Franc", "Brazilian Real",
             "Saudi Riyal", "Shekel"],
        )
        payment_format = st.selectbox(
            "Payment Format",
            ["Cash", "ACH", "Cheque", "Wire", "Credit Card", "Bitcoin"]
        )
        hour = st.slider("Transaction Hour (0–23)", 0, 23, 12)

    with col2:
        amount_paid = st.number_input(
            "Amount Paid", min_value=0.0, value=1000.0, step=0.01,
            help="Amount sent by the source account"
        )
        payment_currency = st.selectbox(
            "Payment Currency",
            ["US Dollar", "Euro", "Bitcoin", "Rupee", "Mexican Peso",
             "UK Pound", "Yen", "Ruble", "Yuan", "Australian Dollar",
             "Canadian Dollar", "Swiss Franc", "Brazilian Real",
             "Saudi Riyal", "Shekel"],
        )
        from_bank_freq = st.number_input(
            "From Bank Frequency (0–1)",
            min_value=0.0, max_value=1.0,
            value=FROM_BANK_FREQ_DEFAULT,
            format="%.6f",
            help="Relative frequency of the sending bank in historical data"
        )
        to_bank_freq = st.number_input(
            "To Bank Frequency (0–1)",
            min_value=0.0, max_value=1.0,
            value=TO_BANK_FREQ_DEFAULT,
            format="%.6f",
            help="Relative frequency of the receiving bank in historical data"
        )

    week = st.selectbox(
        "Day of Week",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: ["Monday","Tuesday","Wednesday","Thursday",
                                "Friday","Saturday","Sunday"][x],
        index=0,
    )

    threshold = st.slider(
        "Decision Threshold",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Lower threshold → more sensitive (catches more laundering but more false positives)"
    )

    submitted = st.form_submit_button("🔍 Analyse Transaction", use_container_width=True)

# ── Prediction ──────────────────────────────────────────────────────────────────
if submitted:
    payment_mismatch = int(receiving_currency != payment_currency)
    amount_diff      = amount_received - amount_paid

    row = pd.DataFrame([{
        "Amount Received":     amount_received,
        "Receiving Currency":  receiving_currency,
        "Amount Paid":         amount_paid,
        "Payment Currency":    payment_currency,
        "Payment Format":      payment_format,
        "Hour":                hour,
        "Week":                week,
        "Payment_mismatch":    payment_mismatch,
        "From_bank_freq":      from_bank_freq,
        "To_Bank_freq":        to_bank_freq,
        "Amount_Diff":         amount_diff,
    }])

    try:
        X_transformed = preprocessor.transform(row)
        prob = float(model.predict(X_transformed, verbose=0)[0][0])
        prediction = int(prob >= threshold)

        st.divider()
        st.subheader("Result")

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.metric("Laundering Probability", f"{prob:.1%}")
        with col_r2:
            st.metric("Amount Difference", f"{amount_diff:,.2f}")

        if prediction == 1:
            st.error(
                f"🚨 **SUSPICIOUS** — This transaction has been flagged as "
                f"potential money laundering (probability: {prob:.1%}).",
                icon="🚨",
            )
        else:
            st.success(
                f"✅ **LEGITIMATE** — This transaction appears normal "
                f"(probability of laundering: {prob:.1%}).",
                icon="✅",
            )

        # Feature summary
        with st.expander("View computed features"):
            st.dataframe(row, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ── Footer ──────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: 2-layer neural network (RMSprop, lr=0.0033) trained on the "
    "LI-Small_Trans dataset with SMOTE oversampling."
)