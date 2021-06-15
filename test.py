import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

df = load_iris()

st.title('Simple Iris Flower **Prediction** App')
st.write('This ap predicts **Iris flower** type')

st.sidebar.header('user input parameters')
def data_input():
    sep_len = st.sidebar.slider('sepal length',4.30,5.40,7.90)
    sep_wid = st.sidebar.slider('sepal width',2.00,4.40,3.40)
    pet_len = st.sidebar.slider('petal legth',1.00,6.90,1.38)
    pet_wid = st.sidebar.slider('[etal width',0.10,2.58,0.20)

    the_data={
        's_len': sep_len,
        's_wid' : sep_wid,
        'p_len' : pet_len,
        'p_wid': pet_wid
    }

    feature = pd.DataFrame(the_data,index=[0])
    return feature


slide = data_input()

st.header('user input parameters.')
st.write(slide)

st.subheader('class label and thier corresponding number.')
st.write(df.target_names)



x = df.data
y = df.target
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
clf = RandomForestClassifier()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)
predict = clf.predict(slide)
proba = clf.predict_proba(slide)

st.subheader('Prediction probablity.')
st.write(proba)

st.subheader('Prediction name. ')
st.write(df.target_names[predict])


