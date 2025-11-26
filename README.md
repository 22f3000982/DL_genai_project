
---

# DL + Gen AI Project (Tentative Title)

**Name:** Ashish  
**Student ID:** 22f3000982  

---

## 📂 Folder Structure


DL_genai_project/
├── scripts/        # Python or helper scripts
├── notebooks/      # Jupyter/Colab notebooks
└── data/           # Datasets (keep large files in .gitignore)


### 📌 Notes
- Repository is private (with instructor access).
- Initial setup contains only folder structure and README.
- Actual code, notebooks, and data will be added later during the project.
w&B token - 9b57595ccb9cc6a50b8e1980f8a6b15d46ec6a71


# Emotion Classification using NLP (Multi-Label Learning)

## 📌 Problem Statement
Given a piece of English text, predict which emotions are present among the following:
`anger, fear, joy, sadness, surprise`

A sample text may belong to **multiple emotions at the same time**, so this is a **multi-label classification task**.

Evaluation Metric → **Macro F1-Score across the 5 labels**

---

## 📂 Models Implemented (as required in L1 Viva)
| Model Type | Model Name | Notes |
|------------|-------------|-------|
| From Scratch | TF-IDF + Logistic Regression | Classical ML baseline |
| Pretrained Transformer | RoBERTa-base | Best performing model |
| Second Transformer | DistilBERT-base-uncased | Lightweight alternative |

All **3 models are tracked using Weights & Biases (wandb)** with metrics such as:
✔ train loss  
✔ validation loss  
✔ Macro F1  
✔ Micro F1  
✔ Accuracy  

---

## 🧠 Project Structure

project_root/
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── model1_scratch_tfidf_logreg.ipynb
│   ├── model2_roberta_multilabel.ipynb
│   ├── model3_distilbert_multilabel.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train_roberta.py
│   ├── train_distilbert.py
│
└── data/
    └── README.txt

