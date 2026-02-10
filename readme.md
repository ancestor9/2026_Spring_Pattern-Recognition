# Syllabus
[Notion](https://www.notion.so/2026_Spring_Pattern-Recognition-2ff49ab460ef80009069cb8fe2935447)
# 딥러닝 기초 및 응용 실라버스 (15주, 주 3시간)

## 1. 교과목 개요
- **교과목명**: 딥러닝 기초 및 응용
- **수업시간**: 주 1회, 3시간 (이론 + 실습)
- **수업방식**: 이론 강의 + Python 기반 실습 (NumPy, PyTorch)
- **선수지식**: Python 기초, 미분 개념(권장), [Scikit-learn Machine Learning](https://github.com/ancestor9/AI-with-python)

## 2. 학습목표
본 교과목은 딥러닝에 필수적인 **선형대수, 통계, 확률 이론**을 이해하고,  
이를 바탕으로 **FNN, CNN, RNN 등 주요 딥러닝 모델의 원리와 구현 능력**을 기르는 것을 목표로 한다.

---

## 3. 주차별 강의 계획

### Week 1. 딥러닝 개요 및 수학적 기초 정리
- 이론
  - 딥러닝의 발전 배경과 활용 분야
  - 머신러닝 vs 딥러닝
  - 딥러닝 학습을 위한 수학 로드맵
- 실습
  - Python / NumPy 환경 점검
  - 벡터·행렬 연산 복습

---

### Week 2. 선형대수 I – 벡터와 행렬
- 이론
  - 벡터, 행렬, 스칼라
  - 내적, 외적, 행렬 곱
- 실습
  - NumPy를 이용한 행렬 연산
  - 브로드캐스팅 이해

---

### Week 3. 선형대수 II – 선형변환과 고유값
- 이론
  - 선형변환의 의미
  - 고유값과 고유벡터
  - 차원 축소 개념(PCA 직관)
- 실습
  - 고유값 분해 실습
  - 간단한 PCA 구현

---

### Week 4. 확률론 기초
- 이론
  - 확률변수와 확률분포
  - 이산/연속 확률분포
  - 기댓값과 분산
- 실습
  - 난수 생성
  - 확률분포 시각화

---

### Week 5. 통계 기초 및 데이터 이해
- 이론
  - 평균, 분산, 공분산
  - 정규분포와 중심극한정리
  - 데이터 정규화의 필요성
- 실습
  - 데이터 스케일링
  - 통계량 계산 실습

---

### Week 6. 미분과 경사하강법
- 이론
  - 미분의 직관적 의미
  - 편미분과 기울기
  - 경사하강법(Gradient Descent)
- 실습
  - 손실함수 시각화
  - 경사하강법 구현

---

### Week 7. 신경망 기초 – 퍼셉트론과 FNN
- 이론
  - 퍼셉트론
  - 다층 신경망(FNN) 구조
  - 활성화 함수
- 실습
  - FNN 직접 구현 (NumPy)
  - 활성화 함수 비교

---

### Week 8. 손실함수와 최적화 기법
- 이론
  - MSE, Cross-Entropy
  - Optimizer (SGD, Momentum, Adam)
  - Overfitting 개념
- 실습
  - PyTorch 기초
  - Optimizer 비교 실험

---

### Week 9. 딥러닝 학습 안정화 기법
- 이론
  - Weight Initialization
  - Regularization (L1, L2)
  - Dropout
- 실습
  - 학습 곡선 분석
  - 과적합 방지 실험

---

### Week 10. CNN I – 합성곱 신경망 개요
- 이론
  - 이미지 데이터 특성
  - Convolution / Pooling
  - CNN 구조
- 실습
  - CNN 기본 구조 구현
  - 이미지 분류 실습

---

### Week 11. CNN II – 심화 및 응용
- 이론
  - 필터와 특징 맵
  - 유명 CNN 구조 (LeNet, AlexNet 개요)
- 실습
  - CIFAR-10 분류
  - 성능 개선 실험

---

### Week 12. RNN I – 시계열과 순환신경망
- 이론
  - 시계열 데이터 특성
  - RNN 구조
  - BPTT 개념
- 실습
  - 간단한 RNN 구현
  - 시계열 예측 실습

---

### Week 13. RNN II – LSTM과 GRU
- 이론
  - Vanishing Gradient 문제
  - LSTM / GRU 구조
- 실습
  - LSTM 기반 시계열 예측
  - 결과 해석

---

### Week 14. 딥러닝 모델 종합 실습
- 이론
  - 모델 선택 전략
  - 하이퍼파라미터 튜닝
- 실습
  - FNN / CNN / RNN 중 하나 선택
  - 미니 프로젝트 진행

---

### Week 15. 프로젝트 발표 및 정리
- 활동
  - 프로젝트 결과 발표
  - 모델 성능 비교
- 정리
  - 딥러닝 핵심 개념 정리
  - 향후 학습 로드맵 제시

---

## 4. 평가 방법 (예시)
- 출석 및 참여: 20%
- 과제 및 실습: 40%
- 기말 프로젝트: 40%

---

## 5. 기대 학습 성과
- 딥러닝에 필요한 수학적 기초 이해
- 주요 딥러닝 모델 구조 설명 가능
- PyTorch 기반 딥러닝 모델 구현 능력 확보


## 6. 관련 자료

[1. Visual explanations of core machine learning concepts](https://mlu-explain.github.io/)

[2. Master AI Concepts with Interactive Learning!](https://www.101ai.net/overview/basics)

[3. Deep-Learning-with-TensorFlow](https://github.com/ancestor9/2025_Winter_Deep-Learning-with-TensorFlow)

[4. Deep-Learning-with-Pytorch](https://github.com/ancestor9/2025_Winter_Deep-Learning-with-Pytorch)
