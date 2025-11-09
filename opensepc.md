
id: email-spam-classification
title: AIoT-DA2025 Email Spam Classification System
version: 1.0.0
status: active
owner: aiot-da2025
maintainers:
  - name: Your Name
    email: your.email@example.com
created: 2025-11-09
updated: 2025-11-09

summary: |
  Build a complete, reproducible Email Spam Classification pipeline including
  data pre-processing, SVM training, metric evaluation, and Streamlit-based visualization UI.

purpose:
  - Develop an offline-friendly spam detector for emails.
  - Demonstrate full ML pipeline: cleaning, tokenization, vectorization, training, evaluation, visualization.
  - Provide an interactive Streamlit interface for real-time spam prediction and metric display.

architecture:
  stack:
    python: ">=3.9, <=3.12"
    node: "LTS (for OpenSpec CLI only)"
    frameworks:
      - scikit-learn
      - pandas
      - numpy
      - matplotlib
      - seaborn
      - joblib
      - streamlit
  components:
    - preprocessing: "Data cleaning, tokenization, TF-IDF vectorization"
    - model: "Support Vector Machine (SVC with linear kernel)"
    - evaluation: "Metric calculation and performance visualization"
    - ui: "Streamlit dashboard for demo and user testing"
  reproducibility: true
  seed: 42
  training_time_limit: 2min
  hardware: "CPU-only"

workflow:
  steps:
    1. Load dataset (CSV with 'label', 'email_text')
    2. Clean text (lowercase, remove punctuation, URLs, stopwords)
    3. Tokenize and vectorize (TF-IDF)
    4. Train SVM classifier (linear kernel, deterministic seed)
    5. Evaluate model (Accuracy, Precision, Recall, F1)
    6. Visualize metrics (Confusion Matrix, ROC curve)
    7. Deploy via Streamlit UI for interactive testing

interfaces:
  cli:
    - name: train_svm.py
      purpose: Train SVM-based spam classifier
      args:
        --input: CSV dataset path
        --output: Path to save model (.joblib)
        --seed: Random seed
        --test-size: Hold-out split ratio
    - name: evaluate.py
      purpose: Compute metrics and save plots
      args:
        --model: Saved model path
        --test-data: CSV test set path
        --output-dir: Directory for charts and reports
    - name: app.py
      purpose: Launch Streamlit dashboard for spam prediction
      args:
        --model: Path to saved model
        --vectorizer: Path to TF-IDF vectorizer
  ui:
    streamlit:
      pages:
        - "Upload Email CSV and classify"
        - "Type or paste email content to predict"
        - "Show confusion matrix, precision-recall, ROC curve"

datasets:
  - name: email_spam_dataset.csv
    format: CSV
    columns:
      - label: spam or ham
      - email_text: message content
    encoding: UTF-8

metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC

visualizations:
  - Confusion Matrix heatmap
  - ROC Curve
  - Precision-Recall Curve
  - Bar chart of evaluation metrics

validation:
  - 80/20 hold-out split
  - Fixed seed reproducibility
  - Manual smoke tests via Streamlit UI

dependencies:
  python-packages:
    - scikit-learn>=1.2
    - pandas>=1.3
    - numpy>=1.21
    - seaborn
    - matplotlib
    - joblib
    - streamlit
    - nltk
  node-tools:
    - openspec-cli (global install)

deliverables:
  - scripts/train_svm.py
  - scripts/evaluate.py
  - app.py (Streamlit UI)
  - models/email_svm_model.joblib
  - visualizations/*.png
  - openspec/changes/add-email-spam-classifier/change.yaml

git:
  branch-policy: main stable, feature via openspec changes
  commit-style: "feat(scope): message [change-id]"
  deterministic: true

acceptance:
  - Model trains under 2 minutes on CPU.
  - F1 ≥ 0.85 on hold-out test set.
  - Streamlit UI runs locally via `streamlit run app.py`.
  - Visualizations display confusion matrix and ROC curve.

