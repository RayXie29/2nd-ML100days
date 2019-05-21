import numpy as np
from sklearn.model_selection import train_test_split, KFold

x = np.arange(50).reshape(10,5)
y = np.zeros(10)

y[:5] = 1
print("Shape of X : ",x.shape)
print(x)
print('-' * 20)
print("Shape of Y : ",y.shape)
print(y)
print('-' * 20)

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = 0.33, random_state = 42)
print(train_x)
print(train_y)
print('-'*20)
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = 0.33)
print(train_x)
print(train_y)
print('-'*20)

#If we didnt set the random state, then the splited out data will have large change that different every time.

kf = KFold(n_splits=5,shuffle = True)
i = 0

for train_index,test_index in kf.split(x):
    i += 1
    train_x,test_x = x[train_index],x[test_index]
    train_y,test_y = y[train_index],y[test_index]
    print("FOLD {} : " .format(i))
    print("train_index : ",train_index)
    print("test_index : ",test_index)
    print("x_test : ",test_x)
    print("y_test : ",test_y)
    print('-' * 30)
    
#If we turn on the shuffle in KFold, the output index will not follow the ascending order
#But every fold in data will still be trained and tested
    
x = np.arange(1000).reshape(200,5)
y = np.zeros(200)
y[:40] = 1
print(y)

trial_times = 1000
random_state = 0
for random_state in range(0,trial_times):
    train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = 20,random_state = random_state)
    if np.sum(test_y) == 10 :
        break

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = 20,random_state = random_state)
print(random_state)
print(test_y)
