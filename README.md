# Logistic Regression
경사하강법(Gradient Descent)을 직접 구현해 보고 이를 활용한 Logistic Regression을 구현해 본다.  
직접 구현해본 코드를 기반으로 Kaggle의 Heart Disease 데이터를 활용해 그 유용성을 확인해 본다.
- [Logistic 함수](#Logistic-Function)
- [오차 역전파](#Propagation)
- [최적화 과정](#Optimize)
###

## Logistic Function
독립변수의 선형 결합을 이용해 어떤 사건의 발생 확률을 예측하는 함수.  
sigmoid함수는 대표적인 logistic 함수로 0~1 값 사이의 미분 가능한 값을 반환하는 특징이 있다.  
이때문에 확률 예측문제에 가장 많이 쓰이며 나아가 분류(classification)문제에 유용하다.
<p align="center"><img src = "https://user-images.githubusercontent.com/67997760/130343858-7006237d-7257-4fd9-8749-c341d445913b.png" width="70%" hedight="70%">

~~~python
def logistic(z):
    return 1/(1+np.exp(-z)) # z = wx + b (w: weight, b: bias, x: data)
~~~
###

## Propagation
학습이 이루어지기 위해선 현재 자신이 가지고있는 가중치가 정답에 근접한지 먼저 판단해야 한다.  
따라서 첫번째 정답 예측값인 Y_hat을 먼저 구해줘야 한다.
~~~python
Y_hat = logistic(np.dot(np.transpose(w),np.transpose(X)) + b)
~~~
그리고 label 데이터(학습 데이터의 라벨값)를 활용해 그 차이를 계산해야 한다.  
이를 비용함수(Cost function)라 하겠다.  
중요한 점은 선형회귀와 같은 비용함수를 사용하면 local optima로 인해 학습이 이루어지지 않는다.  
즉 convex한 함수로 만들어 줘야하는데 여기에 사용되는 방법이 log를 이용하는 방법이다.

<p align="center"><img src = "https://user-images.githubusercontent.com/67997760/130344454-027fc7e8-4dc7-4ffc-bc00-4690946f3fd1.png" width="30%" height="30%">
<p align="center"><img src = "https://user-images.githubusercontent.com/67997760/130345077-01aeaf4b-a3f9-4bcc-ac2a-fd7b542832be.png" width="70%" height="70%">

이를 코드로 나타내면 아래와 같다.
~~~python
eps=1e-5
cost = -1 / m * np.sum(Y * np.log(Y_hat+eps) + (1 - Y) * np.log(1 - Y_hat + eps))
~~~
여기서 eps(epsilon)을 추가한 이유는 혹시나 log연산에 있어서 마이너스 무한대로 수렴했을때 연산오류가 발생하기 때문이다.
이제 cost를 기반으로 오차가 최소값을 가지도록 weight와 bias를 업데이트하는 과정을 확인해 보자.
###

## Optimize 
경사 하강법은 결국 cost function의 기울기를 계산해서 최소 값에 근접하도록 만드는 알고리즘이다.  
여러번의 iteration을 반복하면서 학습률(learning rate)을 기반으로 최소값을 찾아가는 과정을 구현해 본다.
앞서 구현한 오차 전파과정을 기반으로 여러번의 loop를 돌면서 각 가중치의 미분값을 기반으로 갱신한다.
~~~python
# Backward Propagation
dw = 1 / m * np.dot(np.transpose(X), np.transpose(Y_hat - Y))
db = 1 / m * np.sum(Y_hat - Y)

# 각 iteration에서 수행
w = w - learning_rate * dw
b = b - learning_rate * db
~~~
###


