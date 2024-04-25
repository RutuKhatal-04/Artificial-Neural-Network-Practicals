import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
df=load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(df.data,df.target,test_size=0.20,random_state=42)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
model=tf.keras.models.Sequential([tf.keras.layers.Dense(1,activation='sigmoid',input_shape=(X_train.shape[1],))])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=5)
y_pred=model.predict(X_test)
test_loss,test_accuracy=model.evaluate(X_test,y_test)
print("accuracy is",test_accuracy)
