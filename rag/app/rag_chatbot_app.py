#streamlit run ./rag_chatbot_app.py --server.port 8080

import streamlit as st #all streamlit commands will be available through the "st" alias
import requests as req
from typing import Dict


#add page title and configuration
# Page title
st.set_page_config(page_title='Virtual assistant for knowledge base 👩‍💻', layout='wide')
st.title("👩‍💻 Virtual assistant for a knowledge base") #page title
st.subheader(f" Powered by :blue[Bedrock Titan] for text generation and :blue[Bedrock Titan] for embeddings")


api = "https://160wg7g56l.execute-api.us-west-2.amazonaws.com/prod/llm?query="


#input elements
input_text = st.chat_input("Chat with your bot here") #display a chat input box

if input_text: #run the code in this if block after the user submits a chat message
    
    with st.chat_message("user"): #display a user chat message
        st.markdown(input_text) #renders the user's latest message
    
    
    headers: Dict = {"accept": "application/json", "Content-Type": "application/json"}
    
    chat_response = req.get(api,params={'query': input_text},headers=headers,)
    

    with st.chat_message("assistant"): #display a bot chat message
        st.markdown(chat_response.json()["result"]) #display bot's latest response
    
