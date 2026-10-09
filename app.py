import streamlit as st
import joblib
from pathlib import Path
from spam_agent import load_email, classify_text

st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

st.title("📧 Hybrid Spam Detector")
st.caption("Two-stage classifier: TF-IDF + logistic regression gating an LLM.")


@st.cache_resource
def load_model():
    """Load the trained model once and cache it."""
    model_path = Path("spam_model.joblib")
    if not model_path.exists():
        st.error("Model not found. Run `python spam_agent.py train sample_data.csv` first.")
        st.stop()
    return joblib.load(model_path)


pipe = load_model()

# ---------- Input ----------
st.subheader("Enter an email")

subject = st.text_input("Subject", placeholder="Verify your account now")
body = st.text_area("Body", height=200,
                    placeholder="Your account will be suspended. Click here to confirm your password.")

# ---------- Classify button ----------
if st.button("Classify", type="primary", use_container_width=True):
    text = f"{subject}\n{body}".strip()

    if not text:
        st.warning("Please enter a subject or body before classifying.")
    else:
        with st.spinner("Analyzing..."):
            result = classify_text(subject, "", body, pipe=pipe)

        col1, col2 = st.columns(2)
        label_display = result["label"].upper()
        emoji = "🚨" if result["label"] == "spam" else "✅"
        col1.metric("Label", f"{emoji} {label_display}")
        col2.metric("Spam probability", f"{result['confidence']:.1%}")

        st.progress(result["confidence"])
        st.info(f"**Decided by:** {result['stage']}")
        st.write(f"**Reason:** {result['reason']}")


# ---------- Sidebar ----------
with st.sidebar:
    st.header("About")
    st.write(
        "This app classifies emails using a two-stage pipeline:\n\n"
        "1. **Stage 1 (ML)** — a TF-IDF + logistic regression model scores every email.\n"
        "2. **Stage 2 (LLM)** — only borderline emails (probability between 0.10 and 0.90) "
        "are sent to Claude for a nuanced verdict.\n\n"
        "If the LLM is unavailable, the system falls back to the ML score."
    )
    st.divider()
    st.caption("Source: [GitHub](https://github.com/Raj2678/spam-detector)")