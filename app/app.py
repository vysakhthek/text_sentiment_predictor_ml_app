import streamlit as st
import pandas as pd
import numpy as np
import joblib



pipe_log_reg = joblib.load(open("models/sentiment_predictor_pipe_log_reg.pkl", "rb"))

def predict_sentiment(docx):
    results = pipe_log_reg.predict([docx])
    return results[0]

def prediction_probability(docx):
	results = pipe_log_reg.predict_proba([docx])
	return results

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

def main():
    st.title("Text Sentiment Predictor App")
    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.subheader("Home - Sentiment In Text")

        with st.form(key='sentiment_form'):
            raw_txt = st.text_area("Type Here")
            submit_txt = st.form_submit_button(label='submit')
        if submit_txt:
            col1, col2 = st.columns(2)

            prediction = predict_sentiment(raw_txt)
            probability = prediction_probability(raw_txt)

            with col1:
                st.success("Original Text")
                st.write(raw_txt)
                st.success("Prediction")
                st.write(prediction )
            with col2:
                st.success("Prediction probability")
                st.write(probability)
                
 
    elif choice == "Monitor":
        st.subheader("Monitor App")
    
    else:
        st.subheader("About")


if __name__ == '__main__':
    main()                                                                   