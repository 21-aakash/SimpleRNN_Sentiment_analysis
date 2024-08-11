# Step 1: Import Libraries and Load the Model
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')


# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


import streamlit as st
## streamlit app

st.title('Hey ;) this is sky\'s IMDB Movie Review Sentiment Analysis Tool ')
st.write('Enter a movie review to clasify it into: positive / negative')

# User input
user_input = st.text_area('Movie Review: ')


if st.button('Classify'):

    with st.spinner('Wait for it...'):
          time.sleep(2)
  
    
    preprocessed_input=preprocess_text(user_input)
    
    ## MAke prediction
    prediction=model.predict(preprocessed_input)
    sentiment='Positive +' if prediction[0][0] > 0.5 else 'Negative + '

    # Display the result
    
    st.success('This is a success message!', icon="✅")
    st.balloons()
    st.write(f'Sentiment: {sentiment}')
    st.write(f'The Prediction Score: {prediction[0][0]}')
else:
    
    st.warning('Please enter a movie review !!!')


