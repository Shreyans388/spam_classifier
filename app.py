import streamlit as st
import pickle
import nltk
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


pipe=pickle.load(open('pipelineee.pkl','rb'))



st.title("SMS/EMAIL CLASSIFIER")
input1=st.text_area("Enter your message")

if st.button("Classify email"):
    def transform(text_series):
        ps = PorterStemmer()
        processed = []
        for text in text_series:
            text = text.lower()
            tokens = nltk.word_tokenize(text)
            tokens = [word for word in tokens if word.isalnum()]
            tokens = [word for word in tokens
                     if word not in stopwords.words('english')
                     and word not in string.punctuation]
            tokens = [ps.stem(word) for word in tokens]
            processed.append(" ".join(tokens))
        return processed

    input2 = transform([input1])

    y_pred=pipe.predict(input2)[0]

    if y_pred == 1:
        st.write("🚨 Your message is **SPAM**")
    else:
        st.write("✅ Your message is **HAM**")


