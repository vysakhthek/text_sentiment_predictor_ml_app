import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import plotly.express as px 
from datetime import datetime
from track_utils import create_page_visited_table,add_page_visited_details,view_all_page_visited_details,add_prediction_details,view_all_prediction_details,create_emotionclf_table


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
    create_page_visited_table()
    create_emotionclf_table()
    if choice == "Home":
        add_page_visited_details("Home",datetime.now())
        st.subheader("Home - Sentiment In Text")

        with st.form(key='sentiment_form'):
            raw_txt = st.text_area("Type Here")
            submit_txt = st.form_submit_button(label='submit')
        if submit_txt:
            col1, col2 = st.columns(2)

            prediction = predict_sentiment(raw_txt)
            probability = prediction_probability(raw_txt)

            add_prediction_details(raw_txt,prediction,np.max(probability),datetime.now())

            with col1:
                st.success("Original Text")
                st.write(raw_txt)
                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{} : {}".format(prediction,emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))

            with col2:
                st.success("Prediction probability")
                # st.write(probability)
                probability_df = pd.DataFrame(probability,columns=pipe_log_reg.classes_)
                # st.write(probability_df.T)
                probability_df_clean = probability_df.T.reset_index()
                probability_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(probability_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)
                
    elif choice == "Monitor":
        add_page_visited_details("Monitor",datetime.now())
        st.subheader("Monitor App")
        with st.expander("Page Metrics"):
            page_visited_details = pd.DataFrame(view_all_page_visited_details(),columns=['Pagename','Time_of_Visit'])
            st.dataframe(page_visited_details)

            pg_count = page_visited_details['Pagename'].value_counts().rename_axis('Pagename').reset_index(name='Counts')
            c = alt.Chart(pg_count).mark_bar().encode(x='Pagename',y='Counts',color='Pagename')
            st.altair_chart(c,use_container_width=True)

            p = px.pie(pg_count,values='Counts',names='Pagename')
            st.plotly_chart(p,use_container_width=True)
        

        with st.expander('Sentimeent Predictor Metrics'):
            print(view_all_prediction_details)
            df_emotions = pd.DataFrame(view_all_prediction_details(),columns=['Rawtext','Prediction','Probability','Time_of_Visit'])
            st.dataframe(df_emotions)

            prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
            pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction',y='Counts',color='Prediction')
            st.altair_chart(pc,use_container_width=True)

    else:
        st.subheader("About")
        add_page_visited_details("About",datetime.now())
        st.write("Created by Vysakh Thekkath")
        st.markdown("[LinkedIn - vysakh-thekkath](https://www.linkedin.com/in/vysakh-thekkath/)")
        st.markdown("[Github - vysakhthek](https://github.com/vysakhthek)")
        st.write("It's an app that utilizes Natural Language Processing (NLP) techniques to analyze and determine the sentiment of any given text. Whether you want to gauge the sentiment of a social media post, customer review, news article, or any other piece of text, it provides accurate and reliable predictions.")
        st.write("This app can also monitor and track app page usage and the predicted sentiment and probability of each text. It's built using Machine Learning-Natural Language Processing techniques and Streamlit, an open-source app framework for Machine Learning and Data Science teams.")
        st.markdown("""
            - numpy, pandas for Data Analysis
            - neattext for Data Cleaning
            - altair, plotly and seaborn for Data Vizualization
            - scikit-learn for Predictive Analysis
            - streamlit to create custom web apps
            - joblib to provide lightweight pipelining"
            """)
        st.write("This app's model currently has an accuracy of around 70%. I'm currently working on improving the accuracy of the ML model")
        

if __name__ == '__main__':
    main()                                                                   