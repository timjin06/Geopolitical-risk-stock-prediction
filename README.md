# 지정학적 위기 기반 Reddit 감정분석 및 금융시장 예측 프로젝트

## 1. 프로젝트 개요

본 프로젝트는 **지정학적 위기 상황에서 온라인 커뮤니티의 감정 변화가 금융시장 움직임과 연결될 수 있는지**를 분석하는 것을 목표로 한다.  
특히 Reddit 댓글에서 나타나는 공포, 분노, 불안, 기대감 등의 감정 신호를 추출하고, 이를 주가·거래량·시장지표와 결합하여 주가 방향성 예측에 활용하였다.

분석 대상 사건은 다음 두 가지이다.

- **이스라엘-하마스 전쟁**: 2023년 9월 7일 ~ 2023년 11월 7일
- **러시아-우크라이나 전쟁**: 2022년 2월 24일 이후 구간을 외부 테스트 데이터로 활용

전체 과정은 다음과 같이 구성된다.

```text
Step 1. Reddit 텍스트 전처리 및 워드클라우드 분석
Step 2. 감정분석 모델 실행, 비교, 시각화 및 최종 모델 선정
Step 3. 금융·시장 피처 생성, 피처 조합 실험, 하이퍼파라미터 튜닝, 피처 중요도 분석, 외부 테스트 검증
```

---

## 2. 연구 목적

본 프로젝트의 핵심 질문은 다음과 같다.

> 지정학적 위기 발생 시 Reddit 사용자들의 감정 변화가 금융시장 변동성 및 주가 방향성 예측에 유의미한 정보를 제공하는가?

이를 확인하기 위해 다음 절차를 수행하였다.

1. Reddit 댓글 데이터를 수집하고 전처리한다.
2. 여러 감정분석 모델을 적용하여 일별 감정 피처를 생성한다.
3. 감정 피처와 금융·시장 피처를 결합한다.
4. 주가의 다음 거래일 방향성을 `down`, `neutral`, `up`으로 라벨링한다.
5. 다양한 피처 조합 및 모델 설정을 비교한다.
6. 최종 모델에서 어떤 피처가 예측에 기여했는지 Permutation Importance로 해석한다.
7. 이스라엘-하마스 전쟁 구간으로 학습한 모델을 우크라이나-러시아 전쟁 구간에 적용하여 외부 검증을 수행한다.

---

## 3. 전체 파일 구조

```text
.
├── reddit_wordcloud_analysis(step1).ipynb
├── reddit_wordcloud_preprocessed_data.csv
│
├── emotion_analysis_part1(step2).ipynb
├── emotion_analysis_part2(step2).ipynb
│
├── israel_hamas_FIN_MKT_features(step3).csv
├── model_input_dataset_3day_volatility(step3).csv
├── model_input_dataset_5day_volatility(step3).csv
│
├── feature_combination_experiments(step3).ipynb
├── feature_combination_results(step3).csv
├── ablation_timeseries(step3).ipynb
├── hyperparameter_tuning(step3).ipynb
├── permutation_importance(1).ipynb
│
├── ukraine_russia- raw_reddit_testdataset(step3).csv
├── ukraine_goemotions_daily_sentiment_carryforward(step3).csv
├── ukraine_russia_FIN_MKT_features(step3).csv
├── ukraine_final_input_dataset(step3)(1).csv
└── ukraine_model_evaluation(step3).ipynb
```
---

## 4. Step 1: Reddit 텍스트 전처리 및 워드클라우드 분석

### 4.1 사용 파일

| 파일명 | 설명 |
|---|---|
| `reddit_wordcloud_analysis(step1).ipynb` | Reddit 댓글 데이터를 전처리하고 전쟁 전후 워드클라우드를 생성하는 코드 |
| `reddit_wordcloud_preprocessed_data.csv` | 워드클라우드 생성을 위해 전처리된 Reddit 댓글 데이터셋 |

### 4.2 코드 설명

`reddit_wordcloud_analysis(step1).ipynb`에서는 다음 작업을 수행한다.

1. 필요한 라이브러리 설치 및 불러오기
2. Reddit 원본 데이터 로드
3. 작성 시각(`created_time`)을 기준으로 전쟁 전후 구간 분리
4. 텍스트 정제
5. NLTK 기반 토큰화 및 불용어 제거
6. 워드클라우드 생성
7. 전처리된 데이터 저장

전쟁 기준일은 **2023년 10월 7일**로 설정하였다.

```text
pre_war  : 2023-10-07 이전
post_war : 2023-10-07 이후
```

### 4.3 데이터셋 설명

`reddit_wordcloud_preprocessed_data.csv`는 총 **48,485개 댓글**과 **15개 컬럼**으로 구성되어 있다.

주요 컬럼은 다음과 같다.

| 컬럼명 | 설명 |
|---|---|
| `self_text` | Reddit 댓글 원문 |
| `subreddit` | 댓글이 작성된 subreddit |
| `created_time` | 댓글 작성 시각 |
| `author_name` | 작성자명 |
| `score` | 댓글 점수 |
| `post_title` | 댓글이 달린 게시글 제목 |
| `period` | 전쟁 전후 구분 (`pre_war`, `post_war`) |
| `wc_text` | 워드클라우드 생성을 위해 전처리된 텍스트 |

---

## 5. Step 2: 감정분석 및 모델 비교

### 5.1 사용 파일

| 파일명 | 설명 |
|---|---|
| `emotion_analysis_part1(step2).ipynb` | 여러 감정분석 모델을 실행하고 일별 감정 점수를 생성하는 코드 |
| `emotion_analysis_part2(step2).ipynb` | 감정분석 결과를 바탕으로 상관분석, 시각화, 모델 비교 및 최종 모델 선정을 수행하는 코드 |

### 5.2 Part 1: 감정분석 모델 실행

`emotion_analysis_part1(step2).ipynb`에서는 전처리된 Reddit 댓글 데이터(`modified_preprocessing.csv`)를 입력으로 받아 여러 감정분석 모델을 실행한다.

주요 처리 과정은 다음과 같다.

1. `modified_preprocessing.csv` 로드
2. `created_time`을 날짜 형식으로 변환
3. `cleaned_text`를 모델 입력용 텍스트로 사용
4. 너무 긴 텍스트는 앞부분 1,000자 기준으로 제한
5. Hugging Face Transformers 기반 각종 감정 분석 모델 실행
6. 댓글 단위 감정 점수 추출
7. 날짜별 감정 평균값 집계
8. 모델별 daily CSV 파일 저장
9. 결과 파일들을 zip으로 묶어 다운로드

즉, 이 노트북은 **댓글 단위 텍스트를 모델에 입력하여 감정 점수를 계산하고, 이를 날짜별 평균 감정 데이터로 변환하는 단계**이다.

### 5.3 사용한 감정분석 모델

| 모델명 | Hugging Face 모델 ID | 특징 |
|---|---|---|
| BERT GoEmotions | `monologg/bert-base-cased-goemotions-original` | GoEmotions 기반 BERT 모델 |
| DistilBERT GoEmotions | `joeddav/distilbert-base-uncased-go-emotions-student` | 경량화된 GoEmotions 모델 |
| DistilRoBERTa GoEmotions | `SamLowe/roberta-base-go_emotions` | GoEmotions 기반 RoBERTa 계열 모델 |
| ModernBERT | `answerdotai/ModernBERT-base` | ModernBERT 기반 감정분석 후보 모델 |
| Twitter RoBERTa | `cardiffnlp/twitter-roberta-base-sentiment-latest` | 소셜미디어 텍스트에 특화된 RoBERTa 감성 모델 |

### 5.4 Part 2: 상관분석, 시각화 및 모델 선정

`emotion_analysis_part2(step2).ipynb`에서는 모델별 일별 감정 결과를 금융시장 지표와 비교한다.

주요 작업은 다음과 같다.

1. Reddit 댓글 전처리 데이터 생성
2. GoEmotions 기반 감정분석 수행
3. 모델별 daily emotion CSV 로드
4. Part 1 감정 모델 분석 결과와 함께 비교
4. ETF 및 시장 데이터(`final_daily_etf.csv`)와 날짜 기준 병합
5. 감정 지표 구성
6. Spearman 상관분석 수행
7. 감정 지표와 시장 지표의 시계열 시각화
8. 모델별 결과 비교
9. 최종 감정분석 모델 선정

분석에 사용된 대표 감정 지표는 다음과 같다.

| 지표 | 설명 |
|---|---|
| `Panic_Index` | 공포·불안 계열 감정 기반 지표 |
| `Euphoria_Index` | 긍정·기대 계열 감정 기반 지표 |
| `Anger_Index` | 분노 계열 감정 기반 지표 |
| `fear_d1` | 전일 대비 fear 변화량 |
| `Panic_Index_d1` | 전일 대비 Panic Index 변화량 |
| `Euphoria_Index_d1` | 전일 대비 Euphoria Index 변화량 |

---

## 6. Step 3: 금융·시장 피처 생성 및 모델링

## 6.1 이스라엘-하마스 학습 데이터

### 사용 파일

| 파일명 | 설명 |
|---|---|
| `israel_hamas_FIN_MKT_features(step3).csv` | 이스라엘-하마스 전쟁 구간의 기업별 금융 및 시장 피처 |
| `model_input_dataset_3day_volatility(step3).csv` | 3일 변동성 기준 최종 모델 입력 데이터셋 |
| `model_input_dataset_5day_volatility(step3).csv` | 5일 변동성 기준 최종 모델 입력 데이터셋 |

최종모델에는 vol3d => 즉 변동성 피처를 3일로 사용
### `israel_hamas_FIN_MKT_features(step3).csv`

이 파일은 기업별 주가 데이터와 시장 지표를 결합한 FIN/MKT 피처 데이터셋이다.

- 데이터 크기: **5,236행 × 44열**
- 기간: **2023-09-07 ~ 2023-11-07**
- 관측 단위: `Date` × `ticker`

주요 컬럼은 다음과 같다.

| 구분 | 예시 컬럼 | 설명 |
|---|---|---|
| 기본 정보 | `Date`, `sector`, `ticker` | 날짜, 산업군, 기업 티커 |
| 기업 가격 데이터 | `Close`, `Volume` | 종가, 거래량 |
| 수익률 | `return_1d`, `return_3d` | 1일/3일 수익률 |
| 변동성 | `volatility_3d` | 3일 기준 변동성 |
| 시장 지표 | `SPY_return_1d`, `QQQ_return_3d`, `VIX_Close` | 시장 전체 흐름 |
| 원자재·거시 지표 | `Oil_return_1d`, `Gold_return_1d`, `Dollar_return_1d`, `Treasury10Y_return_1d` | 유가, 금, 달러, 10년물 국채금리 관련 지표 |

### 최종 모델 입력 데이터셋

`model_input_dataset_3day_volatility(step3).csv`와 `model_input_dataset_5day_volatility(step3).csv`는 감정 피처, 금융 피처, 시장 피처, 기업 및 산업 one-hot encoding 변수를 결합한 최종 학습용 데이터셋이다.

두 파일 모두 다음 특징을 가진다.

- 데이터 크기: **5,236행 × 172열**
- 기간: **2023-09-07 ~ 2023-11-07**
- 관측 단위: `Date` × `ticker`
- 차이점:
  - `model_input_dataset_3day_volatility(step3).csv`: `volatility_3d`, `relative_volatility_to_sector_3d` 사용
  - `model_input_dataset_5day_volatility(step3).csv`: `volatility_5d`, `relative_volatility_to_sector_5d` 사용

주요 피처 그룹은 다음과 같다.

| 피처 그룹 | 설명 |
|---|---|
| ID | `sector_`, `ticker_`로 시작하는 산업 및 기업 one-hot 변수 |
| FIN | 기업 단위 수익률, 변동성, 거래량 변화, 섹터 대비 상대 수익률 |
| MKT | SPY, QQQ, VIX, Oil, Gold, Dollar, Treasury10Y 등 시장 및 거시 지표 |
| EMO | Reddit 댓글 기반 감정 피처 |
| ECON | Reddit 댓글 내 경제·시장 관련 키워드 비율 |

---

## 6.2 피처 조합 실험

### 사용 파일

| 파일명 | 설명 |
|---|---|
| `feature_combination_experiments(step3).ipynb` | 피처 그룹 조합별 모델 성능을 비교하는 Ablation Study 코드 |
| `feature_combination_results(step3).csv` | 피처 조합 실험 결과 저장 파일 |
| `ablation_timeseries(step3).ipynb` | TimeSeriesSplit 방식으로 피처 조합 실험을 재검증한 코드 |

### 실험 설계

`feature_combination_experiments(step3).ipynb`에서는 다음 조건으로 실험을 수행하였다.

| 항목 | 내용 |
|---|---|
| 고정 피처 | ID 피처 (`sector_`, `ticker_` one-hot encoding) |
| 조합 피처 | FIN, MKT, EMO, ECON |
| 피처 조합 수 | 15가지 |
| FIN 버전 | 3일 변동성 기준, 5일 변동성 기준 |
| 라벨 임계값 | ±0.5%, ±0.7%, ±1.0% |
| 모델 | `HistGradientBoostingClassifier` |
| 교차검증 | 날짜 기준 `GroupKFold` 5-fold |
| Holdout | 마지막 8거래일 |
| 평가 지표 | Macro F1-score |

라벨은 다음 거래일 수익률(`next_return`)을 기준으로 생성하였다.

| 라벨 | 조건 | 의미 |
|---|---|---|
| 0 | `next_return < -threshold` | down |
| 1 | `-threshold <= next_return <= threshold` | neutral |
| 2 | `next_return > threshold` | up |

### 실험 결과 파일

`feature_combination_results(step3).csv`는 총 **90행 × 8열**로 구성되어 있다.

주요 컬럼은 다음과 같다.

| 컬럼명 | 설명 |
|---|---|
| `fin_ver` | 3일 변동성 또는 5일 변동성 기준 |
| `threshold` | 라벨링 임계값 |
| `groups` | 사용한 피처 조합 |
| `n_features` | 사용한 피처 개수 |
| `cv_f1` | 교차검증 Macro F1 |
| `holdout_f1` | Holdout Macro F1 |
| `n_train` | 학습 샘플 수 |
| `n_holdout` | Holdout 샘플 수 |

---

## 6.3 TimeSeriesSplit 기반 Ablation 재검증

`ablation_timeseries(step3).ipynb`는 일반적인 GroupKFold 방식 외에 **시간 순서를 보존하는 TimeSeriesSplit**을 사용하여 피처 조합 실험을 다시 수행한 코드이다.

이 파일의 목적은 다음과 같다.

- 같은 날짜에 여러 기업 데이터가 존재하는 구조를 고려
- 미래 데이터가 과거 예측에 섞이지 않도록 방지
- 실제 금융 예측 상황에 더 가까운 평가 방식 적용
- 과거 → 미래 방향의 검증 구조 유지

주요 설정은 다음과 같다.

| 항목 | 내용 |
|---|---|
| 고정 피처 | ID 피처 |
| 조합 피처 | FIN, MKT, EMO, ECON |
| FIN 버전 | 3일 변동성, 5일 변동성 |
| 라벨 임계값 | ±0.5%, ±0.7%, ±1.0% |
| 모델 | `HistGradientBoostingClassifier` |
| 검증 방식 | 날짜 기준 `TimeSeriesSplit` 3-split |
| 평가 지표 | Macro F1-score |

---

## 6.4 하이퍼파라미터 튜닝

### 사용 파일

| 파일명 | 설명 |
|---|---|
| `hyperparameter_tuning(step3).ipynb` | 최종 후보 모델의 하이퍼파라미터를 탐색하고 Holdout 성능을 평가하는 코드 |

`hyperparameter_tuning(step3).ipynb`에서는 Ablation Study 결과를 바탕으로 최종 후보 설정을 정한 뒤 하이퍼파라미터 튜닝을 수행하였다.

확정된 기본 설정은 다음과 같다.

| 항목 | 내용 |
|---|---|
| 데이터셋 | 3일 변동성 기준 입력 데이터 |
| 피처 조합 | ID + FIN + MKT + EMO |
| 라벨 기준 | ±0.7% |
| 모델 | `HistGradientBoostingClassifier` |
| 평가 방식 | CV로 최적 조합 선택 후 Holdout 1회 평가 |

탐색한 주요 하이퍼파라미터는 다음과 같다.

| 하이퍼파라미터 | 설명 |
|---|---|
| `learning_rate` | 학습률 |
| `max_leaf_nodes` | leaf node 수 |
| `max_iter` | boosting 반복 횟수 |
| `l2_regularization` | L2 정규화 강도 |
| `max_depth` | 트리 깊이 |

튜닝 결과는 `tune_v2_results.csv`로 저장되도록 구성되어 있다.

---

## 6.5 Permutation Importance 기반 피처 중요도 분석

### 사용 파일

| 파일명 | 설명 |
|---|---|
| `permutation_importance(1).ipynb` | 최종 선택된 모델에서 개별 피처가 예측 성능에 얼마나 기여했는지 분석하는 코드 |

`permutation_importance(1).ipynb`는 최종 모델을 단순히 학습·평가하는 데서 끝나지 않고, **어떤 변수가 모델 예측에 실제로 중요한 영향을 미쳤는지 해석하기 위한 파일**이다.

이 노트북의 주요 흐름은 다음과 같다.

1. 최종 모델 입력 데이터셋 로드
2. 3일 변동성 기준 데이터 사용
3. 최종 피처 조합인 ID + FIN + MKT + EMO 구성
4. `HistGradientBoostingClassifier` 학습
5. Holdout 데이터에서 기준 Macro F1-score 계산
6. 각 피처를 하나씩 무작위로 섞어 성능 감소폭 측정
7. 성능이 많이 떨어지는 순서대로 피처 중요도 산출
8. 전체 피처 중요도 CSV 및 상위 중요 피처 그래프 저장

Permutation Importance는 다음과 같은 방식으로 해석할 수 있다.

> 특정 피처를 무작위로 섞었을 때 모델 성능이 크게 하락하면, 해당 피처는 모델 예측에 중요한 변수라고 볼 수 있다.

이 파일의 산출물은 다음과 같다.

| 산출물 | 설명 |
|---|---|
| `feature_importance.csv` | 전체 피처의 Permutation Importance 결과 |
| `top_importance.png` | 상위 중요 피처를 시각화한 그래프 |

따라서 이 파일은 최종 보고서에서 **모델 해석 가능성**, **피처 중요도**, **감정 피처의 기여도**를 설명할 때 사용할 수 있다.

---

## 7. 우크라이나-러시아 외부 테스트 데이터

본 프로젝트는 이스라엘-하마스 전쟁 구간에서 구성한 모델이 다른 지정학적 위기 상황에도 적용 가능한지 확인하기 위해 우크라이나-러시아 전쟁 데이터를 외부 테스트셋으로 사용하였다.

### 7.1 사용 파일

| 파일명 | 설명 |
|---|---|
| `ukraine_russia- raw_reddit_testdataset(step3).csv` | 우크라이나-러시아 전쟁 관련 Reddit 원본 댓글 테스트 데이터 |
| `ukraine_goemotions_daily_sentiment_carryforward(step3).csv` | 우크라이나 댓글의 일별 GoEmotions 감정분석 및 이월 처리 결과 |
| `ukraine_russia_FIN_MKT_features(step3).csv` | 우크라이나-러시아 전쟁 구간의 금융 및 시장 피처 |
| `ukraine_final_input_dataset(step3)(1).csv` | 우크라이나-러시아 외부 테스트용 최종 모델 입력 데이터셋 |
| `ukraine_model_evaluatio(step3).ipynb` | 2022년 우크라이나-러시아 데이터에 대한 외부 테스트 평가 코드 |

### 7.2 Reddit 원본 테스트 데이터

`ukraine_russia- raw_reddit_testdataset(step3).csv`는 우크라이나-러시아 전쟁 관련 Reddit 댓글 원본 데이터이다.

- 데이터 크기: **16,707행 × 5열**
- 날짜 범위: **2021-04-15 ~ 2022-07-25**

주요 컬럼은 다음과 같다.

| 컬럼명 | 설명 |
|---|---|
| `comments` | Reddit 댓글 본문 |
| `date` | 댓글 작성 날짜 |
| `post_id` | 게시글 ID |
| `comment_id` | 댓글 ID |

### 7.3 우크라이나 감정분석 데이터

`ukraine_goemotions_daily_sentiment_carryforward(step3).csv`는 우크라이나-러시아 전쟁 관련 댓글을 GoEmotions 모델로 분석한 뒤, 일별 감정 점수로 집계하고 결측 날짜에 대해 감정값을 이월 처리한 데이터이다.

- 데이터 크기: **15행 × 36열**
- 날짜 범위: **2022-02-24 ~ 2022-04-25**

주요 컬럼은 다음과 같다.

| 컬럼명 | 설명 |
|---|---|
| `date` | 날짜 |
| `fear`, `anger`, `sadness`, `neutral` 등 | GoEmotions 감정별 일별 평균 점수 |
| `comment_count` | 해당 날짜 Reddit 댓글 수 |
| `Panic_Index` | 공포·불안 계열 종합 지표 |
| `Euphoria_Index` | 긍정·기대 계열 종합 지표 |
| `Anger_Index` | 분노 계열 종합 지표 |
| `fear_d1` | 전일 대비 fear 변화량 |
| `Panic_Index_d1` | 전일 대비 Panic Index 변화량 |
| `Euphoria_Index_d1` | 전일 대비 Euphoria Index 변화량 |

### 7.4 우크라이나 FIN/MKT 피처 데이터

`ukraine_russia_FIN_MKT_features(step3).csv`는 우크라이나-러시아 전쟁 기간의 기업별 금융 및 시장 피처 데이터이다.

- 데이터 크기: **4,879행 × 52열**
- 날짜 범위: **2022-02-24 ~ 2022-04-22**
- 관측 단위: `Date` × `ticker`

이스라엘-하마스 데이터와 동일하게 기업 수익률, 변동성, 거래량 변화, 시장 지표, 원자재 및 거시 지표를 포함한다.

### 7.5 우크라이나 최종 입력 데이터셋

`ukraine_final_input_dataset(step3)(1).csv`는 우크라이나 감정 피처와 금융·시장 피처를 결합한 외부 테스트용 최종 입력 데이터셋이다.

- 데이터 크기: **1,428행 × 160열**
- 날짜 범위: **2022-02-24 ~ 2022-03-21**
- 관측 단위: `Date` × `ticker`

주요 피처는 다음과 같다.

| 피처 그룹 | 설명 |
|---|---|
| ID | 산업 및 기업 one-hot encoding |
| FIN | 기업별 수익률, 변동성, 거래량 변화 |
| MKT | 시장 지수, VIX, 유가, 금, 달러, 국채금리 |
| EMO | 우크라이나 Reddit 댓글 기반 감정 피처 |

### 7.6 외부 테스트 평가 코드

`ukraine_model_evaluatio(step3).ipynb`는 2023년 이스라엘-하마스 데이터로 학습한 모델을 2022년 우크라이나-러시아 데이터에 적용하는 Out-of-Sample 평가 코드이다.

주요 설정은 다음과 같다.

| 항목 | 내용 |
|---|---|
| 훈련 데이터 | 2023-09-07 ~ 2023-11-07 이스라엘-하마스 데이터 |
| 테스트 데이터 | 2022-02-24 ~ 2022-03-21 우크라이나-러시아 데이터 |
| 모델 | `HistGradientBoostingClassifier` |
| 주요 하이퍼파라미터 | `learning_rate=0.01`, `max_depth=3`, `max_leaf_nodes=20`, `max_iter=300`, `l2_regularization=0.0` |
| 라벨 기준 | ±0.7% 기준 3분류 |
| 평가 지표 | Macro F1-score, classification report |
| 추가 분석 | 날짜별 Macro F1, 섹터별 Macro F1 |

---

## 8. 모델링 방법

### 8.1 예측 대상

모델은 다음 거래일 주가 수익률을 기준으로 세 가지 클래스를 예측한다.

| 클래스 | 의미 |
|---|---|
| `down` | 다음 거래일 수익률이 음의 임계값보다 낮음 |
| `neutral` | 다음 거래일 수익률이 임계값 범위 안에 있음 |
| `up` | 다음 거래일 수익률이 양의 임계값보다 높음 |

### 8.2 사용 모델

주요 예측 모델은 다음과 같다.

```python
HistGradientBoostingClassifier
```

해당 모델을 사용한 이유는 다음과 같다.

- 다양한 수치형 피처를 처리하기 적합함
- 비선형 관계를 학습할 수 있음
- 피처 조합 실험에서 반복 학습이 가능함
- 금융 데이터처럼 변동성이 큰 데이터에서 비교적 안정적인 성능을 보일 수 있음

### 8.3 평가 지표

본 프로젝트에서는 클래스 불균형 가능성을 고려하여 **Macro F1-score**를 주요 평가 지표로 사용하였다.

Accuracy만 사용할 경우 특정 클래스에 편향된 예측이 과대평가될 수 있기 때문에, `down`, `neutral`, `up` 세 클래스를 균등하게 평가하는 Macro F1-score를 사용하였다.

---

## 9. 설치 방법

### 9.1 저장소 클론

```bash
git clone https://github.com/timjin06/NLP_PROJECY.git
cd NLP_PROJECY
```

### 9.2 가상환경 생성

```bash
python -m venv venv
```

### 9.3 가상환경 활성화

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

### 9.4 필요한 라이브러리 설치

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk wordcloud transformers torch tqdm scipy xgboost lightgbm
```

Jupyter Notebook 실행을 위해 다음도 설치할 수 있다.

```bash
pip install notebook ipykernel
```

---

## 10. 주요 라이브러리

| 구분 | 라이브러리 |
|---|---|
| 데이터 처리 | `pandas`, `numpy` |
| 자연어처리 | `nltk`, `transformers`, `torch` |
| 진행률 표시 | `tqdm` |
| 통계 분석 | `scipy` |
| 시각화 | `matplotlib`, `seaborn`, `wordcloud` |
| 머신러닝 | `scikit-learn`, `xgboost`, `lightgbm` |
| 노트북 실행 | `notebook`, `ipykernel` |

---

## 11. 실행 순서

아래 순서대로 실행하면 프로젝트 전체 흐름을 재현할 수 있다.

### Step 1. 워드클라우드 분석

```text
reddit_wordcloud_analysis(step1).ipynb
```

실행 결과:

```text
reddit_wordcloud_preprocessed_data.csv
```

### Step 2. 감정분석 및 모델 선정

먼저 여러 감정분석 모델을 실행한다.

```text
emotion_analysis_part1(step2).ipynb
```

이후 감정분석 결과를 비교하고 최종 모델을 선정한다.

```text
emotion_analysis_part2(step2).ipynb
```

### Step 3. 모델 입력 데이터 구성

이스라엘-하마스 학습 데이터:

```text
israel_hamas_FIN_MKT_features(step3).csv
model_input_dataset_3day_volatility(step3).csv
model_input_dataset_5day_volatility(step3).csv
```

우크라이나-러시아 외부 테스트 데이터:

```text
ukraine_russia- raw_reddit_testdataset(step3).csv
ukraine_goemotions_daily_sentiment_carryforward(step3).csv
ukraine_russia_FIN_MKT_features(step3).csv
ukraine_final_input_dataset(step3)(1).csv
```

### Step 4. 피처 조합 실험

```text
feature_combination_experiments(step3).ipynb
ablation_timeseries(step3).ipynb
```

실행 결과:

```text
feature_combination_results(step3).csv
```

### Step 5. 하이퍼파라미터 튜닝

```text
hyperparameter_tuning(step3).ipynb
```

### Step 6. 피처 중요도 분석

```text
permutation_importance(1).ipynb
```

실행 결과:

```text
feature_importance.csv
top_importance.png
```

### Step 7. 외부 테스트 평가

```text
ukraine_model_evaluatio(step3).ipynb
```

---

## 12. 프로젝트 흐름 요약

```text
Reddit 댓글 데이터
        │
        ▼
텍스트 전처리
        │
        ├── 워드클라우드 분석
        │
        ▼
감정분석 모델 실행
        │
        ▼
일별 감정 피처 생성
        │
        ▼
감정 모델 비교 및 최종 모델 선정
        │
        ▼
금융·시장 피처 생성
        │
        ▼
감정 피처 + FIN/MKT 피처 병합
        │
        ▼
3일/5일 변동성 기준 모델 입력 데이터셋 생성
        │
        ▼
피처 조합 실험 및 하이퍼파라미터 튜닝
        │
        ▼
최종 모델 피처 중요도 분석
        │
        ▼
주가 방향성 예측 모델 평가
        │
        ▼
우크라이나-러시아 전쟁 데이터로 외부 검증
```

---

## 13. 주의사항

1. 일부 노트북은 Google Colab 환경을 기준으로 작성되어 있으므로, 로컬 환경에서 실행할 경우 파일 경로를 수정해야 할 수 있다.
2. Hugging Face 모델을 처음 실행할 때 모델 다운로드 시간이 오래 걸릴 수 있다.
3. GPU 환경에서 감정분석 모델 실행 속도가 훨씬 빠르다.
4. 파일명에 괄호와 공백이 포함되어 있으므로, 터미널에서 실행할 때는 파일명을 따옴표로 감싸는 것이 좋다.
5. `ukraine_model_evaluatio(step3).ipynb`는 파일명을 `ukraine_model_evaluation(step3).ipynb`로 수정하면 더 명확하다.
6. `permutation_importance(1).ipynb`는 파일명을 `step3_permutation_importance_analysis.ipynb`로 수정하면 더 명확하다.
7. 일부 코드 내부에서는 중간 산출물 파일명이 `final_input_vol3d.csv`, `final_input_vol5d.csv`, `company_with_market_features.csv`처럼 사용될 수 있으므로, 현재 저장소 파일명과 맞지 않는 경우 코드 내 경로를 수정해야 한다.

---

## 14. 결론

본 프로젝트는 Reddit 댓글에서 추출한 감정 신호와 금융시장 피처를 결합하여 지정학적 위기 상황에서의 주가 방향성을 예측하고자 하였다.  

단순히 댓글 감정을 분석하는 데서 끝나지 않고, 감정 피처를 실제 금융 데이터와 결합하여 피처 조합 실험, 하이퍼파라미터 튜닝, 피처 중요도 분석, 외부 위기 사건 테스트까지 수행했다는 점에서 의미가 있다.

특히 이스라엘-하마스 전쟁 데이터를 기반으로 구성한 모델을 우크라이나-러시아 전쟁 데이터에 적용함으로써, 특정 사건에만 국한되지 않는 일반화 가능성을 검토하였다.


