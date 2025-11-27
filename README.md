# Emotion Classification from Text – Kaggle Competition

This repository contains my complete project submission for the 2025 SEP DL–GenAI Course Kaggle Competition. The task was to classify emotions from text into 5 categories:

**anger, fear, joy, sadness, surprise**

I tried multiple models - from a basic NLP model built from scratch to fine-tuned transformer models. This repo has everything: training code, evaluation, inference, and W&B experiment tracking.

## Project Overview

| Requirement | Status |
|------------|--------|
| 3 Unique Models | Completed |
| One model from scratch | TF-IDF + Logistic Regression |
| One pretrained model | RoBERTa Base |
| Third model of choice | DistilBERT Multilabel |
| W&B Tracking | Attached |
| Inference Notebook | Included |
| Report PDF | Included |

## Dataset

Dataset provided through Kaggle course portal:
- `train.csv` → text + 5 emotion labels
- `test.csv` → text (labels to be predicted)

Each emotion is multilabel, meaning a sentence can show multiple emotions at once.

## Folder Structure

```
Emotion-Classification-Kaggle
│── README.md
│── report.pdf
│── requirements.txt
│── data/ (empty – with instructions)
│── models/
│     ├── model1_scratch_tfidf_logreg.ipynb
│     ├── model2_roberta_finetune.ipynb
│     ├── model3_distilbert_multilabel.ipynb
│── inference/
│     ├── inference_roberta.ipynb
│── graphs/
│     ├── comparison_plot.png
```

## Models Used

### Model-1 → TF-IDF + Logistic Regression (Scratch Model)

- Converted text → TF-IDF vectors
- Trained 5 separate one-vs-rest Logistic Regression classifiers
- Fast + lightweight but accuracy was comparatively low
- W&B Run: https://wandb.ai/22f3000982-indian-institute-of-technology-madras/2025-sep-dl-genai-project/runs/soaml5h0?nw=nwuser22f3000982
- Notebook-https://colab.research.google.com/github/22f3000982/DL_genai_project/blob/main/notebooks/model1_scratch_tfidf_logreg.ipynb#scrollTo=x4W4-Wf-LRmH

### Model-2 → Fine-Tuned RoBERTa Base

- Used HuggingFace `roberta-base`
- BCEWithLogitsLoss for multilabel output
- Added scheduler + warmup + gradient clipping
- **Best performing model overall**
-  W&B Run:https://wandb.ai/22f3000982-indian-institute-of-technology-madras/2025-sep-dl-genai-project/runs/k1e5g56a?nw=nwuser22f3000982
-  Notebook - https://colab.research.google.com/drive/1Jpis6OQUtI32_epRLfvCctFIr95c73a5#scrollTo=5Ccv01NZ6XX8

### Model-3 → DistilBERT Multilabel

- Smaller transformer (faster than RoBERTa)
- Performance close to RoBERTa but slightly lower
- W&B Run:https://wandb.ai/22f3000982-indian-institute-of-technology-madras/2025-sep-dl-genai-project/runs/ixcsw28f?nw=nwuser22f3000982
- Notebook-https://colab.research.google.com/drive/1dIYSmXg7CuBNd63T74_57A8f8GitrTr2#scrollTo=rTYnlQhePjPW

## Summary Tavle

| Model | Macro-F1 | Micro-F1 | Accuracy | Notes |
|-------|----------|----------|----------|-------|
| TF-IDF + Logistic Regression | 0.67 | 0.75 | 0.53 | Scratch baseline |
| DistilBERT | 0.80 | 0.82 | 0.59 | Good trade-off b/w speed + accuracy |
| **RoBERTa (Best)** | **0.86** | **0.87** | **0.71** | ❗ Final submission model |

### W&B Comparison Graph

![Model Comparison](graphs/comparison_plot.png)

## Learning / Mistakes / Experience

- Fine-tuning transformers was more tricky than expected, especially getting the scheduler and gradient clipping right
- Initially my models were overfitting a LOT; later early stopping fixed it
- W&B really helped in debugging and comparing models side by side
- In future, I want to try ensembling and proper hyper-parameter search



Thanks for reviewing my project

---



**Ashish Raj**    
22f3000982
