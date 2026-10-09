# Hybrid Spam Detector

A two-stage email spam classifier: a TF-IDF + logistic regression model handles the clear-cut cases, and only borderline emails are escalated to an LLM for a verdict and short explanation.

## Quick Start

    pip install -r requirements.txt
    python spam_agent.py train sample_data.csv
    python spam_agent.py check examples/phishing_example.txt

## How it works

- Stage 1 (ML): scores every email with a TF-IDF + logistic regression model.
- Stage 2 (LLM): only emails with spam probability between 0.10 and 0.90 are sent to Claude.
- If the API is unavailable, the system falls back to the ML score.

## Results

Trained on the SMS Spam Collection (5,572 messages).
spam       0.96      0.92      0.94       149

    accuracy                           0.98      1115
   macro avg       0.97      0.96      0.96      1115
weighted avg       0.98      0.98      0.98      1115

## License

MIT
