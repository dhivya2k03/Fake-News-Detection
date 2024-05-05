import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm 
# Load the trained model
lgb=pickle.load(open('lgbmodekl.pkl','rb'))  # Replace 'your_model.pkl' with the path to your trained model file
# Define the Streamlit UI
st.header("Fake News Detection")
inp = st.text_area("Enter the news you saw:", height=400, placeholder='Copy & Paste the News here..', max_chars=6000)
# Make prediction when 'Submit' button is clicked
if st.button("Submit"):
    if inp:
        # Vectorize the example sentence
        # vect=pickle.load(open('vector.pkl','rb'))

        # inp_vec = vect.transform([inp])
        #lgbmodel= lgb.LGBMClassifier()
        Tfidfvectorizer = pickle.load(open("TfidfVectorizer.pkl", "rb"))
        inp = Tfidfvectorizer.transform(np.reshape(np.array(inp), -1))
        prediction = lgb.predict(inp)
        if prediction == 1:
            st.error(" FAKE NEWS! ❌ ")
        else:
            st.success("REAL NEWS! ✅")
    else:
        st.warning("Please enter a news article.")
