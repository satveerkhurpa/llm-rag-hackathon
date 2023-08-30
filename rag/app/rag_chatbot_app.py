#streamlit run ./rag_chatbot_app.py --server.port 8080

"""
A simple web application to implement a chatbot. This app uses Streamlit 
for the UI and the Python requests package to talk to an API endpoint that
implements text generation and Retrieval Augmented Generation (RAG) using LLMs
and Amazon OpenSearch as the vector database.
"""

import streamlit as st #all streamlit commands will be available through the "st" alias
import requests as req
from typing import Dict, List
import boto3

# utility functions
def get_cfn_outputs(stackname: str) -> List:
    cfn = boto3.client('cloudformation')
    outputs = {}
    for output in cfn.describe_stacks(StackName=stackname)['Stacks'][0]['Outputs']:
        outputs[output['OutputKey']] = output['OutputValue']
    return outputs

#Get the API Gateway endpoint from the Cloudformation stack.
CFN_STACK_NAME: str = "llm-rag-hackathon"
outputs = get_cfn_outputs(CFN_STACK_NAME)
api = outputs.get("LLMAppAPIEndpoint") + "/llm?query="

####################
# Streamlit code
####################

#add page title and configuration
# Page title
st.set_page_config(page_title='Virtual assistant for knowledge base 👩‍💻', layout='wide')
st.title("👩‍💻 Virtual assistant for a knowledge base") #page title
st.subheader(f" Powered by :blue[Bedrock Titan] for text generation and :blue[Bedrock Titan] for embeddings")

#input elements
input_text = st.chat_input("Chat with your bot here") #display a chat input box

if input_text: #run the code in this if block after the user submits a chat message
    
    with st.chat_message("user"): #display a user chat message
        st.markdown(input_text) #renders the user's latest message
    
    
    headers: Dict = {"accept": "application/json", "Content-Type": "application/json"}
    
    chat_response = req.get(api,params={'query': input_text},headers=headers,)
    

    with st.chat_message("assistant"): #display a bot chat message
        st.markdown(chat_response.json()["result"]) #display bot's latest response
    
