import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

df = pd.read_csv ('heart.csv',encoding='utf-8')
print(df.head())

X =df[['age','sex','cp','trtbps','chol','fbs','restecg','exng','oldpeak','slp','caa','thall']]
dim=X.shape[1]
X=X.to_numpy()
Y= df[['output']] # 2D
Y=Y.to_numpy()
t1=Y.shape[0]
Y=np.reshape(Y,(t1,))  # 2D 轉 1D
x_train , x_test , y_train , y_test = train_test_split(X,Y,test_size=0.02)


y_train2=tf.keras.utils.to_categorical(y_train)
print(y_train2.shape)
category=2      
y_test2=tf.keras.utils.to_categorical(y_test, num_classes=(category))

# 建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=90,
    activation=tf.nn.relu,
    input_dim=dim))
model.add(tf.keras.layers.Dense(units=100,
    activation=tf.nn.relu ))
model.add(tf.keras.layers.Dense(units=category,
    activation=tf.nn.softmax ))
model.compile(optimizer='adam',
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=['accuracy'])
model.fit(x_train, y_train2,
          epochs=170,
          batch_size=80)

#測試
model.summary()

score = model.evaluate(x_test, y_test2 )
print("score:",score)

predict2 = model.predict_classes(x_test)
print("predict_classes:",predict2)
print("y_test",y_test[:])