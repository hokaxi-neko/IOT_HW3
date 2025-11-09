# 📧 Email Spam Classification Web App

這是一個基於 **Streamlit** 的 **Email/SMS Spam 分類系統**，可以直接在 Web 上完成：

- 無需 CLI，即可上傳 CSV 訓練模型  
- 支援 **無表頭、雙引號逗號分隔 CSV**  
- SVM 模型訓練、TF-IDF 向量化  
- 顯示模型評估指標：Accuracy、Precision、Recall、F1  
- 混淆矩陣、ROC 曲線、Precision-Recall 曲線可視化  
- 即時單封郵件預測  

---

## 📝 專案結構

email-spam-classification/
│
├─ app.py # Streamlit Web App 主程式
├─ models/ # 訓練後的模型與向量器
│ ├─ email_svm_model.joblib
│ └─ tfidf_vectorizer.joblib
├─ requirements.txt # Python 依賴套件
└─ README.md


---

## ⚡ 安裝與啟動

1. 建議使用 **Python 3.9+**，並建立虛擬環境：

```bash
pip install -r requirements.txt

    啟動 Web App：

streamlit run app.py

    瀏覽器開啟 http://localhost:8501

即可使用。