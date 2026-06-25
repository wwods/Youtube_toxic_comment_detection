# YouTube Toxic Comment Detection — TF-IDF + BERT Multi-label Classification

영어 YouTube 댓글 1,000건을 대상으로 유해성을 탐지하는 ML/DL 분류 모델.  
전통적인 TF-IDF 기반 모델과 BERT 파인튜닝을 비교하고, 단일 라벨·다중 라벨 분류를 모두 실험한다.

## Overview

| 실험 | 모델 | 방식 |
|---|---|---|
| 단일 라벨 (IsToxic) | Naive Bayes / SVM / Random Forest | TF-IDF 벡터화 |
| 단일 라벨 (IsToxic) | BERT 파인튜닝 | Hugging Face Transformers |
| 다중 라벨 (8개 범주) | Random Forest | TF-IDF 벡터화 |
| 다중 라벨 (8개 범주) | BERT 파인튜닝 | Hugging Face Transformers |

다중 라벨 범주: `IsToxic, IsAbusive, IsThreat, IsProvocative, IsObscene, IsHatespeech, IsRacist, IsNationalist`

## Results

| 모델 | F1-score | Accuracy |
|---|---|---|
| Naive Bayes (TF-IDF) | 0.68 | 0.70 |
| BERT 파인튜닝 | **0.9137** | **0.92** |

## Features

- 라벨별 유해성 비율 분포 시각화
- 라벨 간 상관관계 Heatmap
- BERT Attention Weight 시각화 (판단 근거 해석)
- 모델별 F1-score 비교 차트

## Tech Stack

- Python, scikit-learn, Pandas, Seaborn
- Hugging Face Transformers, TensorFlow, tf-keras
- Google Colab

## Dataset

`youtoxic_english_1000.csv` — 영어 YouTube 댓글 1,000건, 8개 유해성 라벨

## Course

한신대학교 머신러닝 이해와 활용 (ML_2601) 기말 프로젝트
