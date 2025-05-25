import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

data=pd.read_csv("personality_dataset.csv")

for i in range(len(data)):
  if data.loc[i,'Stage_fear']=="Yes":
    data.loc[i,'Stage_fear']=data.loc[i,'Stage_fear']=1
  elif data.loc[i,'Stage_fear']=="No":
    data.loc[i,'Stage_fear']=data.loc[i,'Stage_fear']=0
for i in range(len(data)):
  if data.loc[i,'Drained_after_socializing']=="Yes":
    data.loc[i,'Drained_after_socializing']=data.loc[i,'Drained_after_socializing']=1
  elif data.loc[i,'Drained_after_socializing']=="No":
    data.loc[i,'Drained_after_socializing']=data.loc[i,'Drained_after_socializing']=0
data=data.drop_duplicates()
data=data.dropna()
x=data.iloc[:,:7]
y=data.iloc[:,7:]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
model=LogisticRegression()
y_train=np.ravel(y_train)
model.fit(x_train,y_train)
res=accuracy_score(y_test,model.predict(x_test))
print(res)