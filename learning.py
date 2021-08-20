import model as ml
import pandas as pd

# load data(path)
data = pd.read_csv("data/CriteoSearchData", sep="\t", header=None, nrows=1500000)

# Sales를 제외한 Outcome/labels 제거
data = data.drop([1,2], axis=1)

# nb_click_1week, product_price 만 학습데이터로 결정
data = data.drop([i for i in range(6,23)],axis=1)
data = data.drop([3],axis=1)

# 100만개의 학습 데이터, 50만개의 평가 데이터
train_data = data.iloc[:1000000,:]
test_data = data.iloc[1000000:,:]

# column 이름 정해주기
c_names = ['Sale','nb_click_1week','product_price']
train_data.rename(columns= {0: c_names[0],4:c_names[1],5:c_names[2]}, inplace= True)
test_data.rename(columns= {0: c_names[0],4:c_names[1],5:c_names[2]}, inplace= True)

# nb_click_1week 의 결측 값, -1인 것은 학습에서 제외하기.
filter_ = train_data.nb_click_1week!=-1
train_data = train_data.loc[filter_, :]

# 학습 데이터 준비하기
X = train_data.iloc[:, [1,2]].to_numpy()
Y = train_data.Sale.to_numpy().reshape(1,X.shape[0])

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
