# DACON-Competition1 : 한국어 문서 추출요약 AI경진대회 

![image](https://user-images.githubusercontent.com/75110162/103287720-9b523200-4a26-11eb-8cf4-b9416009727e.png)

처음 참가해보는 DACON Competiton, 좋은 성적을 거두진 못했지만 기록해보기 

## 모델1. TextRANK
CONCEPT: 한국어 Glove 모델을 이용하여 문장들을 임베딩 한 뒤, 임베딩 결과로 RANK를 메겨 상위 3개의 문장을 해당 문서의 추출요약으로 선정 



#### STEP1 : 불용어 제거 
