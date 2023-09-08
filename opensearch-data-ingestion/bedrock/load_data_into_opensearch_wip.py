%%writefile container/load_data_into_opensearch.py

import os
import sys

# this is needed because the credentials.py and sm_helper.py
# are in /code directory of the custom container we are going 
# to create for Sagemaker Processing Job
sys.path.insert(1, '/code')

import glob
import time
import json
import os
import sys
import boto3
import logging
import argparse
import numpy as np
import multiprocessing as mp
from itertools import repeat
from functools import partial
import sagemaker, boto3, json
from typing import List, Tuple
from sagemaker.session import Session
from opensearchpy.client import OpenSearch
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.llms.sagemaker_endpoint import ContentHandlerBase
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from requests_aws4auth import AWS4Auth
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy import OpenSearch, helpers
from opensearchpy.helpers import bulk

# global constants
MAX_OS_DOCS_PER_PUT = 500
TOTAL_INDEX_CREATION_WAIT_TIME = 60
PER_ITER_SLEEP_TIME = 5
logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s,%(module)s,%(processName)s,%(levelname)s,%(message)s', level=logging.INFO, stream=sys.stderr)


def check_if_index_exists(opensearch_domain_endpoint, index_name, region, host, http_auth) -> OpenSearch:
    
    aos_client = OpenSearch(
        hosts = [{'host': opensearch_domain_endpoint.replace("https://", ""), 'port': 443}],
        http_auth = http_auth,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection
    )
    exists = aos_client.indices.exists(index_name)
    logger.info(f"index_name={index_name}, exists={exists}")
    return exists


def get_opensearch_cluster_client(opensearch_domain_endpoint, http_auth, name):
    opensearch_client = OpenSearch(
        hosts = [{'host': opensearch_domain_endpoint.replace("https://", ""), 'port': 443}],
        http_auth = http_auth,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection
    )
    return opensearch_client


def put_bulk_in_opensearch(records_with_embedding, client):
    logging.info(client)
    logging.info(f"Putting {len(records_with_embedding)} documents in OpenSearch")
    success, failed = bulk(client, records_with_embedding)
    return success, failed

def create_vector_embedding_with_bedrock(text, name, bedrock_client):
    payload = {"inputText": f"{text}"}
    body = json.dumps(payload)
    modelId = args.embeddings_model_endpoint_name
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_client.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    embedding = response_body.get("embedding")
    return {"_index": name, "text": text, "vector_field": embedding}

    
def process_shard(shard, embeddings_model_endpoint_name, aws_region, os_index_name, os_domain_ep, os_http_auth) -> int: 
    logger.info(f'Starting process_shard of {len(shard)} chunks.')
    st = time.time()
    embeddings = bedrock_embeddings
    
    # docsearch = OpenSearchVectorSearch(index_name=os_index_name,
    #                                    embedding_function=embeddings,
    #                                    opensearch_url=os_domain_ep,
    #                                    http_auth=os_http_auth,
    #                                    is_aoss=False,
    #                                    use_ssl = True,
    #                                    verify_certs = True,
    #                                    connection_class = RequestsHttpConnection)    
    # docsearch.add_documents(documents=shard)
    
    opensearch_client = get_opensearch_cluster_client(os_domain_ep, os_http_auth, os_index_name)
   
    for chunk in shard:
        records_with_embedding = create_vector_embedding_with_bedrock(chunk, args.opensearch_index_name, boto3_bedrock)
        logger.info(f"Embedding for chunk created")
        #logging.info(records_with_embedding)

        # Bulk put all records to OpenSearch
        success, failed = put_bulk_in_opensearch(records_with_embedding, opensearch_client)
        logging.info(f"Documents saved {success}, documents failed to save {failed}")    


    et = time.time() - st
    logger.info(f'Shard completed in {et} seconds.')
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opensearch-cluster-domain", type=str, default=None)
    parser.add_argument("--opensearch-secretid", type=str, default=None)
    parser.add_argument("--opensearch-index-name", type=str, default=None)
    parser.add_argument("--aws-region", type=str, default="us-west-2")
    parser.add_argument("--embeddings-model-endpoint-name", type=str, default=None)
    parser.add_argument("--chunk-size-for-doc-split", type=int, default=500)
    parser.add_argument("--chunk-overlap-for-doc-split", type=int, default=30)
    parser.add_argument("--input-data-dir", type=str, default="/opt/ml/processing/input_data")
    parser.add_argument("--process-count", type=int, default=1)
    parser.add_argument("--create-index-hint-file", type=str, default="_create_index_hint")
    parser.add_argument("--bedrock-endpoint-url", type=str, default=None)
    parser.add_argument("--llm-model-id", type=str, default=None)
    args, _ = parser.parse_known_args()

    logger.info("Received arguments {}".format(args))
    # list all the files
    files = glob.glob(os.path.join(args.input_data_dir, "*.*"))
    logger.info(f"there are {len(files)} files to process in the {args.input_data_dir} folder")
    
    loader = PyPDFDirectoryLoader(path=args.input_data_dir, silent_errors=True)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "], #split chunks at (1) paragraph, (2) line, (3) sentence, or (4) word, in that order
        chunk_size=args.chunk_size_for_doc_split,
        chunk_overlap=args.chunk_overlap_for_doc_split,
        length_function=len,
    )
    
    # Stage one: read all the docs, split them into chunks. 
    st = time.time() 
    logger.info('Loading documents ...')
    docs = loader.load()
    
    # avg_doc_length = lambda docs: sum([len(doc.page_content) for doc in docs])//len(docs)
    # avg_char_count_pre = avg_doc_length(docs)
    # avg_char_count_post = avg_doc_length(docs)
    # print(f'Average length among {len(docs)} documents loaded is {avg_char_count_pre} characters.')
    # print(f'After the split we have {len(docs)} documents more than the original {len(files)}.')
    # print(f'Average length among {len(docs)} documents (after split) is {avg_char_count_post} characters.')
    
    # add a custom metadata field, such as timestamp
    for doc in docs:
        doc.metadata['timestamp'] = time.time()
        doc.metadata['embeddings_model'] = args.embeddings_model_endpoint_name
    chunks = text_splitter.create_documents([doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs])
    et = time.time() - st
    logger.info(f'Time taken: {et} seconds. {len(chunks)} chunks generated') 
    
    
    db_shards = (len(chunks) // MAX_OS_DOCS_PER_PUT) + 1
    logger.info(f'Loading chunks into vector store ... using {db_shards} shards')
    st = time.time()
    shards = np.array_split(chunks, db_shards)
    
    t1 = time.time()
    
    #OpenSearch Auth
    service = 'es'
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, args.aws_region, service, session_token=credentials.token)
    
    #Bedrock embeddings
    BEDROCK_ENDPOINT_URL = args.bedrock_endpoint_url
    BEDROCK_REGION = args.aws_region
  
    boto3_bedrock = boto3.client(
         service_name='bedrock',
         region_name=BEDROCK_REGION,
         endpoint_url=BEDROCK_ENDPOINT_URL
    )
    
    model_kwargs =  { 
        "maxTokenCount": 8192, 
        "stopSequences": [], 
        "temperature": 0, 
        "topP": 1 
    }


    llm = Bedrock(model_id=args.llm_model_id, client=boto3_bedrock, model_kwargs=model_kwargs)
    bedrock_embeddings = BedrockEmbeddings(client=boto3_bedrock)
    opensearch_domain_with_url = f"https://{args.opensearch_cluster_domain}"
    
    # first check if index exists, if it does then call the add_documents function
    # otherwise call the from_documents function which would first create the index
    # and then do a bulk add. Both add_documents and from_documents do a bulk add
    # but it is important to call from_documents first so that the index is created
    # correctly for K-NN
    logger.info(f'Checking if index exists')
    index_exists = check_if_index_exists(opensearch_domain_with_url,
                                         args.opensearch_index_name,
                                         args.aws_region,
                                         args.opensearch_cluster_domain,
                                         awsauth)
    
    embeddings =  bedrock_embeddings
    
    
    if index_exists is False:
        # create an index if the create index hint file exists
        path = os.path.join(args.input_data_dir, args.create_index_hint_file)
        if os.path.isfile(path) is True:
            logger.info(f"index {args.opensearch_index_name} does not exist but {path} file is present so will create index")
            
                        
            # by default langchain would create a k-NN index and the embeddings would be ingested as a k-NN vector type
            docsearch = OpenSearchVectorSearch.from_documents(index_name=args.opensearch_index_name,
                                                              documents=shards[0],
                                                              embedding=embeddings,
                                                              opensearch_url=opensearch_domain_with_url,
                                                              use_ssl = True,
                                                              verify_certs = True,  
                                                              timeout = 300,
                                                              connection_class = RequestsHttpConnection,
                                                              http_auth=awsauth)
            # we now need to start the loop below for the second shard
            shard_start_index = 1  
        else:
            logger.info(f"index {args.opensearch_index_name} does not exist and {path} file is not present, "
                        f"will wait for some other node to create the index")
            shard_start_index = 0
            # start a loop to wait for index creation by another node
            time_slept = 0
            while True:
                logger.info(f"index {args.opensearch_index_name} still does not exist, sleeping...")
                time.sleep(PER_ITER_SLEEP_TIME)
                index_exists = check_if_index_exists(opensearch_domain_with_url,
                                                     args.opensearch_index_name,
                                                     args.aws_region,  
                                                     args.opensearch_cluster_domain,
                                                     awsauth)
                if index_exists is True:
                    logger.info(f"index {args.opensearch_index_name} now exists")
                    break
                time_slept += PER_ITER_SLEEP_TIME
                if time_slept >= TOTAL_INDEX_CREATION_WAIT_TIME:
                    logger.error(f"time_slept={time_slept} >= {TOTAL_INDEX_CREATION_WAIT_TIME}, not waiting anymore for index creation")
                    break
                
    else:
        logger.info(f"index={args.opensearch_index_name} does exist, going to call add_documents")
        shard_start_index = 0
        
    with mp.Pool(processes = args.process_count) as pool:
        results = pool.map(partial(process_shard,
                                   embeddings_model_endpoint_name=args.embeddings_model_endpoint_name,
                                   aws_region=args.aws_region,
                                   os_index_name=args.opensearch_index_name,
                                   os_domain_ep=args.opensearch_cluster_domain,
                                   os_http_auth=awsauth),
                           shards[shard_start_index:])
    
    t2 = time.time()
    logger.info(f'run time in seconds: {t2-t1:.2f}')
    logger.info("all done")