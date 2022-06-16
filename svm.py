from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt

df = pd.read_csv('dataset/processed_data.csv')

# df = df.info()

# print(df)

# print("데이터프레임의 shape : ",df.shape)
# train, test = train_test_split(df, test_size=0.3)
# train, val = train_test_split(train, test_size=0.3)

#df= df.set_index('SEX_CCD').stack().to_frame().reset_index().rename(columns={'level_1': 'variable', 0: 'value'})

df = df.values

data = np.zeros((len(df[:]), 2))
data[:, 0] = df[:, 0]
data[:, 1] = df[:, 2]

x_train, x_test, y_train, y_test = train_test_split(data, df[:,1], train_size=0.8)

#x_train = np.expand_dims(x_train, axis=2)

s = svm.SVC(gamma=0.001)
s.fit(x_train, y_train)

res = s.predict(x_test)

print('예측: ', res)

scores = cross_val_score(s, x_train, y_train, cv = 5)
scores

pd.DataFrame(cross_validate(s, x_train, y_train, cv = 5))

print('교차검증 평균:' , scores.mean())


plt.scatter(x_train[:,0],x_train[:,1],c=y_train)
plt.xlabel("Store Code")
plt.ylabel("Count of Receipt")
plt.show()


#print(s.fit)



# conf = np.zeros((10,10))
# for i in range(len(res)):
#     conf[res[i]][y_test[i]] += 1
# print(conf)



    


