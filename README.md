# 4× Super Resolution



## 🔹 프로젝트 구조

```
datasets/
    dataloader.py
    imagedataset.py
models/
    swinir.py
utils/
    data_utils.py
    train.py
    test.py
    evaluate.py
    transforms.py
    visualize.py
main.py
requirements.txt
Dockerfile
```
---
## 🔹 실행법 (for Local PC)

### ☝️ Requirements 설치

```
pip install -r requirements.txt
```

### ✌️ 실행

```
python main.py
```
---
## 🔹 도커 실행법 (for Docker User)

### ☝️ 도커 이미지 빌드

```
docker build -t sr_swinir .
```

### ✌️ 컨테이너 실행

```
docker run --rm --gpus all sr_swinir
```

---

## 🔹 데이터셋

Unsplash  
KiTS23 데이터셋은 3D CT 의료 영상으로 총 489명의 환자의 데이터를 포함하고 있습니다. 이 데이터셋은 신장, 종양 부위를 분할하는 데 적합합니다.
  
데이터셋은 공식 깃허브`https://github.com/neheller/kits23`에서 제공되며, 본 도커 이미지 실행시 자동으로 다운로드됩니다.

---

## 🔹 결과

50 에포크 학습 후, 결과:

```
Test Dice Score: 72.96%
```

예측 결과 예시:

![샘플 예측 결과1](images/work4_con1.png)  
![샘플 예측 결과2](images/work4_con2.png)  
![샘플 예측 결과3](images/work4_con3.png)

---
소개 페이지: 
06unoh