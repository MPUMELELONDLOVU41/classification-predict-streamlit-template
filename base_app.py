"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
from PIL import Image

# Vectorizer
news_vectorizer = open("resources/TfidfVec.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# Dictionary of climate change classes
sentiment = {2: "(News) : the tweet links to factual news about climate change",
	1: "(Pro) : the tweet supports the belief of man-made climate change",
	0:"(Neutral) : the tweet neither supports nor refutes the belief of man-made climate change",
	-1:"(Anti) : the tweet does not believe in man-made climate change"}

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information","EDA","Model"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		info_file = open("resources/info.md","r")
		st.markdown(info_file.read())

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user inputv
		tweet_text = st.text_area("Enter Text","Type Here")

		opt = ["Logistic Regression", "Linear SVC","Naive Bayes","Random Forest"]
		models = st.sidebar.selectbox("Choose Model to predict with", opt)

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			# predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			# prediction = predictor.predict(vect_text)

			
			if models == "Logistic Regression":
				predictor = joblib.load(open(os.path.join("resources/log_reg.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

			if models == "Linear SVC":
				predictor = joblib.load(open(os.path.join("resources/svc.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

			if models == "Naive Bayes":
				predictor = joblib.load(open(os.path.join("resources/bayes.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

			if models == "Random Forest":
				predictor = joblib.load(open(os.path.join("resources/forest.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(sentiment[prediction[0]]))
	
	# Building out the EDA page
	if selection == "EDA":
		st.info("Explanatory Data Analysis")
		# You can read a markdown file from supporting resources folder
		eda_file = open("resources/EDA.md","r",)
		st.markdown(eda_file.read())
		image = Image.open('resources/imgs/sentiments.png')
		st.image(image, caption='Sentiments',use_column_width=True)
		image = Image.open('resources/imgs/common.png')
		st.image(image, caption='Sentiments',use_column_width=True)
		image = Image.open('resources/imgs/wordcloud.png')
		st.image(image, caption='Sentiments',use_column_width=True)

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	
	# Building out the Model page
	if selection == "Model":
		st.info("Model Selection and Preprocessing")
		# You can read a markdown file from supporting resources folder
		model_file = open("resources/model.md","r")
		st.markdown(model_file.read())


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
