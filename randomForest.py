import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

#PAGE_CONFIG = {"page_title":"StColab.io","page_icon":":smiley:","layout":"centered"}
#st.beta_set_page_config(**PAGE_CONFIG)

def user_input_features() :
  sepal_length = st.sidebar.slider('sepal_length',4.3, 7.9, 5.4)
  sepal_width = st.sidebar.slider('sepal_width',2.0, 4.4, 3.4)
  petal_length = st.sidebar.slider('petal_length',1.0, 6.9, 1.3)
  petal_width = st.sidebar.slider('petal_width',0.1, 2.5, 0.2)
  data = {'sepal_length' : sepal_length,
          'sepal_width' : sepal_width,
          'petal_length' : petal_length,
          'petal_width' : petal_width
          }
  features = pd.DataFrame(data, index=[0])
  return features

def main():
	#st.title("Awesome Streamlit for MLDDD")
	#st.subheader("How to run streamlit from colab")
  st.write("""
  # Simple Iris Flower Prediction WebApp

  This app predicts the **Iris flower** type!
  
  """)

  st.sidebar.header('User Input Parameters')

  df= user_input_features()

  st.subheader("파라미터를 설정해주세요.")
  st.write(df)

  iris = datasets.load_iris()
  x=iris.data
  y=iris.target

  clf = RandomForestClassifier()
  clf.fit(x,y)

  predict_ = clf.predict(df)
  predict_proba = clf.predict_proba(df)

  st.subheader("Iris 종류 ")
  st.write(iris.target_names)

  st.subheader("예측된 꽃종류")
  st.write(iris.target_names[predict_])

  st.subheader("예측된 꽃종류2")
  st.write(predict_)

  st.subheader("예측된 꽃종류3")
  st.write(iris.target_names)


  st.subheader("꽃종류별 예측 확률")
  st.write(predict_proba)








if __name__ == '__main__':
	main()