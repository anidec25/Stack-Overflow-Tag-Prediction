import pandas as pd
import numpy as np
import pickle
import streamlit as st
import string
import re
from PIL import Image


import nltk

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer




import warnings

import sklearn
warnings.filterwarnings('ignore')
from PIL import Image

# combine the post and title 
# pre-process the data: lowercase, stopwords, stemmming or lemitization, removal of code, whiteapces etc
# vectorize the input
# predict the output
# display



#Page Icon
page_icon = Image.open('./images/logo.png')

#Page config setup
st.set_page_config(
page_title="Stackoverflow Tag Prediction",
page_icon=page_icon,
layout="wide"
)



stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

def text_splitter(text):
    return text.split()

model = pickle.load(open('models/model.pkl','rb'))
tfidf = pickle.load(open('models/vectorizer.pkl','rb'))

# Load and display the image
image = Image.open('./images/orange.png')
new_image = image.resize((1000, 325))
st.image(new_image,caption='')

# Title of the page
st.title('Stack Overflow Tag Prediction')

# User prompt
title = st.text_input('Enter the title of the Question')
title=title.encode('utf-8')
post = st.text_area('Enter the post')

# Concatinating title and post as question.
question = str(title) + " " + str(post)


def striphtml(data):
    """
    Function to clean the html part of the body
    """
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr,' ',str(data))
    return cleantext

def pre_process(question):
    """
    Function to pre-process the text data: 
    Action performed: removing code, striping html, lower, stemming, stopwords removal.
    """

    question=re.sub('<code>(.*?)</code>', '', question, flags=re.MULTILINE|re.DOTALL)
    question=striphtml(question.encode('utf-8'))
    question=re.sub(r'[^A-Za-z]+',' ',question)
    words=word_tokenize(str(question.lower()))

    #Removing all single letter and and stopwords from question exceptt for the letter 'c'
    question=' '.join(str(stemmer.stem(j)) for j in words if j not in stop_words and (len(j)!=1 or j=='c'))

    return question

output_list = ['c#','java','php','javascript','android','jquery','c++','python','iphone','asp.net','mysql','html','.net','ios','objective-c','sql','css','linux','ruby-on-rails','windows']



if st.button('Predict Tags'):

    preprocessed = pre_process(question)

    preprocessed = [word for word in preprocessed.split()]

    vector_input = tfidf.transform(preprocessed)

    y_pred = model.predict(vector_input)



    freq_dict = {}
    words = np.where(y_pred.toarray() == 1)


    for items in words:
        if len(items) == 0:
            result = 'No Tags'
        else:
            if output_list[items[0]] in freq_dict:
                freq_dict[output_list[items[0]]] += 1
            else: 
                freq_dict[output_list[items[0]]] = 1



    sorted_items = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
    # Get the top 3 elements from the sorted list
    top_3_elements = sorted_items[:3]
        
    if len(freq_dict) == 0: 
        st.write(result)
    else:
        col1, col2, col3 = st.columns([1,1,1])

        with col1:
            st.code(top_3_elements[0][0])

        with col2:
            if len(top_3_elements) > 1:
                st.code(top_3_elements[1][0])
            else:
                pass

        with col3:
            if len(top_3_elements) > 2:
                st.code(top_3_elements[2][0])
            else:
                pass

    
    
    

        





