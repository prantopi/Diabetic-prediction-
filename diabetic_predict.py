import pandas as pd
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("diabetic.csv")
#df=df.iloc[:len(df)//1]
df["male"]=df["gender"]=="male"
df["smoke"]=df["smoking_history"].map({"never":0,"current":1,"former":2,"No Info":3,"ever":4,"not current":5})

X=df[["male","age","hypertension","heart_disease","bmi","HbA1c_level","blood_glucose_level","smoke"]].values
df=df.dropna()

y=df["diabetes"].values

model = LogisticRegression(max_iter=5000)
model.fit(X,y)
#print(model.coef_,model.intercept_)


y_pred = model.predict(X)
print((model.score(X, y))*100)
q=model.predict(X[50:100])
r=y[50:100]
print(" Predict:{0}|\nDiagnosis{1}".format(q,r))
