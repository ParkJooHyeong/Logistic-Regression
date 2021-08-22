import model as ml
import pandas as pd

# load data(path)
data = pd.read_csv("data/framingham.csv")


# model 학습 시키기
model = ml.model(X,Y,num_iterations=5000,learning_rate=0.01,print_cost=True)

# 평가 데이터 준비하기
tX = test_data.iloc[:, [1,2]].to_numpy()
tY = test_data.Sale.to_numpy().reshape(1,tX.shape[0])

# 결과 확인하기
print("각 데이터별 에측값 :", ml.predict(model,tX))
print("각 데이터별 예측 확률값 :", ml.predict_prob(model, tX))
print("전체 데이터 accuracy :",ml.score(model,tX,tY))


# 모델 정보 저장하기
# ml.saveModel(model)

# 모델 정보 불러오기
# model_load = ml.loadModel()
