import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

st.title("Diabetes Prediction Using Machine Learning :boy::syringe: ")
st.write("""
This app predicts Diabetes using Machine Learning. 

**Scroll down to see the predictions** :point_down:
""")
data = pd.read_csv('data/diabetes.csv')
st.subheader("Dataset Overview")
st.dataframe(data)
st.subheader("Some statistical informations")
st.write(data.describe())
st.subheader("Data Vizualiation")
st.line_chart(data)

st.sidebar.header("User input parameters")

X = data.iloc[:,0:8].values
Y = data.iloc[:,-1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.25,random_state=0)

def user_input():
    pregnancies = st.sidebar.slider("Pregnancies",0,17,3)
    glucose = st.sidebar.slider("Glucose",0,199,117)
    blood_pressure = st.sidebar.slider("Blood Pressure",0,122,72)
    skin_thickness = st.sidebar.slider("Skin Thickness",0,99,23)
    insulin = st.sidebar.slider("Insulin",0.0,846.0,30.0)
    BMI = st.sidebar.slider("BMI",0.0,67.1,32.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function",0.78,2.42,0.3725)
    age = st.sidebar.slider("Age",21,81,29)

    user_data={'Pregnancies':pregnancies,
               'Glucose':glucose,
               'Blood Pressure':blood_pressure,
               'Skin Thickness':skin_thickness,
               'Insulin':insulin,
               'BMI':BMI,
               'DPF':dpf,
               'Age':age}
    features = pd.DataFrame(user_data,index=['inputs'])
    return features
df = user_input()

st.subheader("User Input Parameters")
st.write(df)

clf = RandomForestClassifier()
clf.fit(X_train,Y_train)

prediction = clf.predict(df)
pred_prob = clf.predict_proba(df)

st.subheader("Prediction")
st.write("0 = **No Diabetes detected** & **Diabtese detected** = 1 ")
st.write(prediction)

st.subheader("Prediction Probability :chart_with_upwards_trend: ")
st.write(pred_prob)

predictions = clf.predict(X_test)
st.subheader("Model Accuracy Score :thumbsup:")
st.write(metrics.accuracy_score(Y_test,predictions))