import pickle

from flask import Flask,render_template, request
from sklearn.preprocessing import StandardScaler
#model.pkl -trained ml model

#Deserialize - read the binary file-Ml model
clf=pickle.load(open('model.pkl','rb'))

###############################################################################################
#for getting range decided on xtrain-repeat the Ml steps till Normalization again
import pandas as pd
df=pd.read_csv("../Web_App_Malaria-outbreak/outbreak_detect.csv")

import math as m
median_maxTemp=df['maxTemp'].median()
print(m.floor(median_maxTemp))

median_minTemp=df['minTemp'].median()
print(m.floor(median_minTemp))

df['maxTemp']=df['maxTemp'].fillna(median_maxTemp)
df['minTemp']=df['minTemp'].fillna(median_minTemp)

from sklearn import preprocessing

#label encoding
LE=preprocessing.LabelEncoder()

#fitting the technique to dataset
df.Outbreak=LE.fit_transform(df.Outbreak) #converting variables to numeric values
df.head(26)

df=df.drop(25,axis=0)
df=df.drop(['Positive',   'pf'],axis =1)

X=df.iloc[:,:-1].values #iloc==>index location 2D array #independent variables x(all col except outbreak)
Y=df.iloc[:,-1:].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

################################################################################################


app=Flask(__name__)

@app.route("/")
def hello():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict_class():
    print([x for x in request.form.values()])
    features=[int(x) for x in request.form.values()]
    print(features)
    sst=StandardScaler().fit(X_train)

    output=clf.predict(sst.transform([features]))
    print(output) #list format
    if(output[0]) == 0:
        return render_template('index.html',pred=f'There is no chance to get malaria')
    else:
        return render_template('index.html',pred=f'There is a chance to get malaria')

if __name__ == "__main__":
    app.run(debug=True)

