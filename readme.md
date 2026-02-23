## 실라버스
## 1. 과목 명 : 데이터표준화
<img src="https://miro.medium.com/v2/resize:fit:1400/1*FTv0YjReFNoE8phRcfFKBg.png" width=600>

## 2. 학습목표
본 교과목은 딥러닝에 필수적인 **선형대수, 통계, 확률 이론**을 이해하고,  
이를 바탕으로 **FNN, CNN, RNN 등 주요 딥러닝 모델의 원리와 구현 능력**을 기르는 것을 목표로 한다.

---

### 3. 수업 진행 및 평가 방식
- 평가는 출석(20%), 개인 혹은 팀평가(30%, 50%)하여 학점 부여 방식 (Quiz, Peer Review 등 점수 반영)
-     github를 개인별로 반드시 만들고 GitHubDeskTop을 설치하여 수업종료시 공개하여야 한다.


## 4. 주차별 강의 계획

| Week | 주제                  | 이론                                                       | 실습                              |
| ---- | ------------------- | -------------------------------------------------------- | ------------------------------- |
| 1    | 통계 기초 및 데이터 이해      | 평균, 분산, 공분산<br>정규분포와 중심극한정리<br>데이터 정규화 필요성               | 데이터 스케일링<br>기초 통계량 계산           |
| 2    | 확률론 기초              | 확률변수와 확률분포<br>이산/연속 확률분포<br>기댓값과 분산                      | 난수 생성<br>확률분포 시각화               |
| 3    | 딥러닝 개요 및 수학적 기초     | 딥러닝 발전 배경 및 활용 분야<br>머신러닝 vs 딥러닝<br>수학 로드맵 정리            | Python / NumPy 복습<br>벡터·행렬 연산   |
| 4    | 선형대수 I – 벡터와 행렬     | 벡터, 행렬, 스칼라<br>내적, 외적<br>행렬 곱                            | NumPy 행렬 연산<br>브로드캐스팅 이해        |
| 5    | 선형대수 II – 선형변환과 고유값 | 선형변환 개념<br>고유값·고유벡터<br>PCA 직관                            | 고유값 분해 실습<br>간단한 PCA 구현         |
| 6    | 미분과 경사하강법           | 미분의 직관적 의미<br>편미분과 기울기<br>경사하강법                          | 손실함수 시각화<br>Gradient Descent 구현 |
| 7    | 신경망 기초 – 퍼셉트론과 FNN  | 퍼셉트론 구조<br>다층 신경망(FNN)<br>활성화 함수                         | FNN 직접 구현 (NumPy)<br>활성화 함수 비교  |
| 8    | 중간고사 및 손실함수와 최적화           | MSE, Cross-Entropy<br>SGD, Momentum, Adam<br>Overfitting | PyTorch 기초<br>Optimizer 비교 실험   |
| 9    | 학습 안정화 기법           | Weight Initialization<br>L1/L2 Regularization<br>Dropout | 학습 곡선 분석<br>과적합 방지 실험           |
| 10   | CNN I – 합성곱 신경망     | 이미지 데이터 특성<br>Convolution / Pooling<br>CNN 구조            | CNN 기본 구조 구현<br>이미지 분류 실습       |
| 11   | CNN II – 심화 및 응용    | 필터와 특징 맵<br>LeNet, AlexNet 개요                            | CIFAR-10 분류<br>성능 개선 실험         |
| 12   | RNN I – 시계열과 순환신경망  | 시계열 데이터 특성<br>RNN 구조                                     | 간단한 RNN 구현<br>시계열 예측            |
| 13   | RNN II – LSTM과 GRU  | Vanishing Gradient 문제<br>LSTM / GRU 구조                   | LSTM 기반 예측<br>결과 해석             |
| 14   | 딥러닝 모델 종합 실습        | 모델 선택 전략<br>하이퍼파라미터 튜닝                                   | FNN / CNN / RNN 중 선택<br>미니 프로젝트 |
| 15   | 프로젝트 발표 및 정리        | 모델 성능 비교<br>딥러닝 핵심 개념 정리                                 | 프로젝트 발표<br>향후 학습 로드맵 정리         |


## 7. 관련 자료

[1. Visual explanations of core machine learning concepts](https://mlu-explain.github.io/)

[2. Master AI Concepts with Interactive Learning!](https://www.101ai.net/overview/basics)

[3. Deep-Learning-with-TensorFlow](https://github.com/ancestor9/2025_Winter_Deep-Learning-with-TensorFlow)

[4. Deep-Learning-with-Pytorch](https://github.com/ancestor9/2025_Winter_Deep-Learning-with-Pytorch)
