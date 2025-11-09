import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

# -------------------------------
# 初始化 NLTK stopwords
# -------------------------------
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

# -------------------------------
# 文本清理
# -------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    return " ".join(tokens)

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Email Spam Classification Web", layout="wide")
st.title("📧 Email/SMS Spam Classification (Web Training + Prediction)")

# -------------------------------
# 側邊欄設定
# -------------------------------
st.sidebar.header("⚙️ 訓練設定")
test_size = st.sidebar.slider("測試集比例", 0.1, 0.5, 0.2)
seed = st.sidebar.number_input("隨機種子", value=42, step=1)
train_button = st.sidebar.button("開始訓練模型")

# 固定資料路徑
dataset_path = "datasets/sms_spam_no_header.csv"
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "email_svm_model.joblib")
vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")

# -------------------------------
# 訓練模型
# -------------------------------
if train_button:
    if os.path.exists(dataset_path):
        # 讀 CSV，無表頭，自動解析雙引號逗號分隔
        df = pd.read_csv(dataset_path, header=None, encoding='utf-8')
        if df.shape[1] < 2:
            st.error("CSV 必須至少有兩欄：label 與 email_text")
        else:
            st.write("前五列資料預覽：")
            st.write(df.head())

            # 清理文本
            st.info("🧹 清理文本...")
            df['clean_text'] = df[1].apply(clean_text)

            # 切分訓練/測試集
            st.info("🔤 切分訓練/測試集...")
            X_train, X_test, y_train, y_test = train_test_split(
                df['clean_text'], df[0], test_size=test_size,
                random_state=seed, stratify=df[0]
            )

            # TF-IDF 向量化
            st.info("⚙️ TF-IDF 向量化...")
            vectorizer = TfidfVectorizer(max_features=5000)
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            # SVM 訓練
            st.info("🛠 訓練 SVM 模型...")
            model = SVC(kernel='linear', random_state=seed, probability=True)
            model.fit(X_train_vec, y_train)
            st.success("🎯 模型訓練完成！")

            # 儲存模型與向量器
            joblib.dump(model, model_path)
            joblib.dump(vectorizer, vectorizer_path)
            st.info(f"💾 模型儲存於：{model_path}")
            st.info(f"💾 向量器儲存於：{vectorizer_path}")

            # 評估指標
            y_pred = model.predict(X_test_vec)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, pos_label='spam')
            rec = recall_score(y_test, y_pred, pos_label='spam')
            f1 = f1_score(y_test, y_pred, pos_label='spam')
            st.subheader("📊 評估指標")
            st.metric("Accuracy", f"{acc:.4f}")
            st.metric("Precision", f"{prec:.4f}")
            st.metric("Recall", f"{rec:.4f}")
            st.metric("F1-score", f"{f1:.4f}")

            # 混淆矩陣
            cm = confusion_matrix(y_test, y_pred, labels=['ham','spam'])
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham','spam'], yticklabels=['ham','spam'])
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)

            # ROC & PR 曲線
            y_true_bin = np.where(y_test=='spam',1,0)
            y_score = model.decision_function(X_test_vec)

            # ROC
            fpr, tpr, _ = roc_curve(y_true_bin, y_score)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
            plt.plot([0,1],[0,1],'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            st.pyplot(plt)

            # Precision-Recall
            precision_vals, recall_vals, _ = precision_recall_curve(y_true_bin, y_score)
            plt.figure()
            plt.plot(recall_vals, precision_vals)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            st.pyplot(plt)
    else:
        st.error(f"找不到資料檔案：{dataset_path}")

# -------------------------------
# 單封郵件即時預測
# -------------------------------
st.subheader("✉️ 單封郵件即時預測")
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # 下拉選擇示範
    example_texts = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts",
        "Hey, are we still meeting for lunch tomorrow?",
        "Congratulations! You won a prize, claim now!"
    ]
    user_input = st.selectbox("選擇範例郵件或自行輸入：", ["手動輸入"] + example_texts)
    if user_input == "手動輸入":
        user_input = st.text_area("輸入郵件內容", height=150)

    if st.button("預測郵件", key="predict_button"):
        if user_input.strip():
            vec = vectorizer.transform([user_input])
            pred = model.predict(vec)[0]
            prob = model.decision_function(vec)[0]
            st.success(f"分類結果：{'🟥 Spam' if pred=='spam' else '🟩 Ham'} (決策值：{prob:.3f})")
        else:
            st.warning("請輸入郵件文字")
else:
    st.info("請先訓練模型")
