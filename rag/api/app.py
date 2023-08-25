from typing import Union
from fastapi import FastAPI
from mangum import Mangum
from fastapi.responses import JSONResponse
import uvicorn
import os
import logging

import boto3
import logging, sys
from langchain.vectorstores import OpenSearchVectorSearch
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from requests_aws4auth import AWS4Auth
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from opensearchpy import RequestsHttpConnection
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationalRetrievalChain


logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger()

api_stage = os.environ.get("API_STAGE", "")
logger.info(f"api_stage={api_stage}")


app = FastAPI(
    root_path=f"{api_stage}",
    docs_url="/docs",
    title="QnA app",
    description="QnA app",
    version="0.0.1",
)

handler = Mangum(app)

@app.get("/")
def read_root():
    return {"API to query bedrock"}

@app.get("/llm", name="query", tags=["Query endpoint"])
def query_llm(query: Union[str, None] = None):
    print(f"api_stage={api_stage}")
    response = {"result" : "Please enter your query in a query string."}
    if query:
        answer = get_answer_using_query(query)
        response = JSONResponse({"result": answer})
        logger.info(response)

    return response
    
@app.get("/ping", name="Healthcheck", tags=["Healthcheck"])
async def healthcheck():
    print(f"api_stage={api_stage}")
    print(f"success: pong!")
    return {"success": "pong!"}
    
    
def get_vectorstore(bedrock_embeddings):
    service = 'es'
    credentials = boto3.Session().get_credentials()
    aws_region = os.environ.get("BWB_REGION_NAME")
    awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, aws_region, service, session_token=credentials.token)
    
    #Bedrock embeddings
    opensearch_index=os.environ.get("opensearch_index")
    opensearch_domain_endpoint=os.environ.get("opensearch_domain_endpoint")
    
    docsearch = OpenSearchVectorSearch(index_name=opensearch_index,
                                   embedding_function=bedrock_embeddings,
                                   opensearch_url=opensearch_domain_endpoint,
                                   http_auth=awsauth,
                                   is_aoss=False,
                                   use_ssl = True,
                                   verify_certs = True,
                                   connection_class = RequestsHttpConnection) 
    return docsearch
    
    
def get_llm_details():
    aws_region = os.environ.get("BWB_REGION_NAME")
    BEDROCK_ENDPOINT_URL = os.environ.get("BWB_ENDPOINT_URL")
    BEDROCK_REGION = aws_region
    
    boto3_bedrock = boto3.client(
        service_name='bedrock',
        endpoint_url=BEDROCK_ENDPOINT_URL,
        region_name=BEDROCK_REGION,
    )
    
    model_kwargs =  { 
        "maxTokenCount": 1024, 
        "stopSequences": [], 
        "temperature": 0, 
        "topP": 0.9 
    }
    
    
    llm = Bedrock(model_id="amazon.titan-tg1-large", client=boto3_bedrock, model_kwargs=model_kwargs)
    bedrock_embeddings = BedrockEmbeddings(client=boto3_bedrock)
    return llm, bedrock_embeddings
    

def get_answer_using_query(query):
    llm, bedrock_embeddings = get_llm_details()
    vectorstore = get_vectorstore(bedrock_embeddings)
    
    wrapper_store = VectorStoreIndexWrapper(vectorstore=vectorstore)
    answer = wrapper_store.query(question=query, llm=llm)
    print('question: ', query)
    print('answer: ', answer)
    
    return answer
    


if __name__ == "__main__":
   uvicorn.run(app, host="0.0.0.0", port=8080)
   
#uvicorn app:app --reload --port 8080