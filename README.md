# DACON-Competition1 : 한국어 문서 추출요약 AI경진대회 

![image](https://user-images.githubusercontent.com/75110162/103287720-9b523200-4a26-11eb-8cf4-b9416009727e.png)

처음 참가해보는 DACON Competiton, 좋은 성적을 거두진 못했지만 기록해보기

![image](https://user-images.githubusercontent.com/75110162/103288247-cf7a2280-4a27-11eb-823a-511ea18cb1bd.png)

article_original 에서 3개의 extractive 문장 선택 

## 모델1. TextRANK
CONCEPT: 한국어 Glove 모델을 이용하여 문장들을 임베딩 한 뒤, 임베딩 결과로 RANK를 메겨 상위 3개의 문장을 문서의 추출요약으로 선정 

#### STEP1 : 전처리
![image](https://user-images.githubusercontent.com/75110162/103288527-634bee80-4a28-11eb-85fc-a453fb24ca8b.png)
한글 이외의 corpus 제거 

#### STEP2-1 : 토큰화
![image](https://user-images.githubusercontent.com/75110162/103288595-8a0a2500-4a28-11eb-98e4-02c9f621d33b.png)
OKT 클래스로 토큰화

#### STEP2-2 : 불용어 제거 
![image](https://user-images.githubusercontent.com/75110162/103288741-f1c07000-4a28-11eb-9809-88529cd6a2dc.png)
외부데이터 한국어 불용어 사전을 이용하여 불용어 제거 

#### STEP3 : 임베딩 
![image](https://user-images.githubusercontent.com/75110162/103288831-22a0a500-4a29-11eb-8aed-a4df52fa2492.png)
한국어 버전의 Glove 임베딩 import

![image](https://user-images.githubusercontent.com/75110162/103288904-495edb80-4a29-11eb-87b4-3913180c5551.png)
문장에 존재하는 단어들의 임베딩을 합하여 문장벡터를 만듦

#### SETP4: RANK 
![image](https://user-images.githubusercontent.com/75110162/103289040-9e9aed00-4a29-11eb-9dff-48e63ffd5857.png)

문서 내의 문장벡터 간의 Cosine Similarity를 구하여 Similariy Matrix를 만든 후, networkx library를 활용하여 문장들 사이의 RANK 결정 
  - 문장들의 임베딩을 기반으로 인접행렬 구성 후 그래프로 표현, 이 후 그래프의 edge weight를 이용하여 각 문장의 score 결정

#### 결과 
순위권 밖의 좋지 못한 score를 기록

#### 원인 분석
- 전처리 미흡
  - 한글 이외의 문자를 모두 제거하면 문장에 아무것도 남지 않는 경우 존재
  - EDA를 통하여 자주 등장하는 단어를 고려하여 불용어를 지정했어야 하는 아쉬움
- 문장벡터 생성 과정에서 문장 내의 단어들의 임베딩을 단순 합하여 문장 벡터를 만드는 것의 논리성 부족 
- Pre-Trained Model을 이용하지 않음

## 모델2. BERT-Extractive-Summarizer
CONCEPT: [BERT-Extractive-Summarizer](https://github.com/dmmiller612/bert-extractive-summarizer)

BERT 임베딩 결과에 K-Means Cluster을 적용하여 각 군집의 중심 문장을 선택 

![image](https://user-images.githubusercontent.com/75110162/103290600-5087e880-4a2d-11eb-8def-d4fd8e11b713.png)

#### STEP1 : bert-extractive-summarizer, KoBERT install
![image](https://user-images.githubusercontent.com/75110162/103291284-b88afe80-4a2e-11eb-98cc-8a8389030720.png)

#### STEP2 : Summarizer 에 custom model, custom tokenizer 적용
![image](https://user-images.githubusercontent.com/75110162/103291407-fd169a00-4a2e-11eb-8812-db963ef34684.png)

이 과정에서 SPACY에 한국어 모델이 없어 Mecab 을 이용하였고.. 이 과정에서 난항을 겪었고 해결하는데 엄청 오래 걸렸다.. 

#### 결과 
20위 까지 순위를 끌어올렸지만 최종적으로는 많은 사람들에게 뒤쳐저 48위까지 떨어지게 되었다

#### 원인 분석
- Fine Tuning 을 할 수 없었다.
  -  bert-extractive-summarizer 가 공식적으로는 fine tuning 기능이 없었다
- 알 수 없는 문장 갯수 오류
  - 추출을 3개 하도록 지정했지만 1개 또는 2개가 추출되는 경우 발생.. 아마 군집 중심값이 존재하지 않는 문제이지 않을까? Github issue에서도 명쾌한 답변이 존재하지 않았다.
  
