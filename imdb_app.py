import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

model = tf.keras.models.load_model('simple_rnn_imdb.h5')
word_index = imdb.get_word_index()

def encode_data (input_data):
    word = input_data.lower().split()
    encoded = [word_index.get(words,2) + 3 for words in word]
    padded_review = pad_sequences([encoded], maxlen=500)
    return padded_review

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):

    preprocessed_input=encode_data(user_input)

    ## MAke prediction
    prediction=model.predict(np.array(preprocessed_input))
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')



