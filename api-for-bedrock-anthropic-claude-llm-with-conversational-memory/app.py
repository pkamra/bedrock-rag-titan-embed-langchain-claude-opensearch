import json
import io
import boto3
import json
import os
import sys
import numpy as np
import time
import re

from langchain.vectorstores import OpenSearchVectorSearch

# We will be using the Titan Embeddings Model to generate our Embeddings.
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

# turn verbose to true to see the full logs and documents
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.schema import BaseMessage
from langchain.prompts import PromptTemplate
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory

from flask import Flask, request, jsonify
from flask_cors import CORS

from utils import bedrock, print_ww  #these utility functions are in the /utils folder

module_path = ".."
sys.path.append(os.path.abspath(module_path))
print_ww(CONDENSE_QUESTION_PROMPT.template)

# We are also providing a different chat history retriever which outputs the history as a Claude chat (ie including the \n\n)
_ROLE_MAP = {"human": "\n\nHuman: ", "ai": "\n\nAssistant: "}
def _get_chat_history(chat_history):
    buffer = ""
    for dialogue_turn in chat_history:
        if isinstance(dialogue_turn, BaseMessage):
            role_prefix = _ROLE_MAP.get(dialogue_turn.type, f"{dialogue_turn.type}: ")
            buffer += f"\n{role_prefix}{dialogue_turn.content}"
        elif isinstance(dialogue_turn, tuple):
            human = "\n\nHuman: " + dialogue_turn[0]
            ai = "\n\nAssistant: " + dialogue_turn[1]
            buffer += "\n" + "\n".join([human, ai])
        else:
            raise ValueError(
                f"Unsupported chat history format: {type(dialogue_turn)}."
                f" Full chat history: {chat_history} "
            )
    return buffer

# the condense prompt for Claude
condense_prompt_claude = PromptTemplate.from_template("""{chat_history}

Answer only with the new question.


Human: How would you ask the question considering the previous conversation: {question}


Assistant: Question:""")

boto3_bedrock = bedrock.get_bedrock_client(
    region=os.environ.get("AWS_DEFAULT_REGION", None)
)

# - create the Anthropic Model
llm = Bedrock(
    model_id="anthropic.claude-v2", client=boto3_bedrock, model_kwargs={"max_tokens_to_sample": 100000}
)
bedrock_embeddings = BedrockEmbeddings(client=boto3_bedrock)

#Initialize the OpenSearchVectorSearch from the already created VectorDatabase

service = 'aoss'
credentials = boto3.Session().get_credentials()
auth = AWSV4SignerAuth(credentials, os.environ.get("AWS_DEFAULT_REGION", None), service)
aoss_host = os.environ.get("AOSS_HOST", None) 

docsearch = OpenSearchVectorSearch(opensearch_url=aoss_host,
index_name="digital-assistant-rag-index",
http_auth=auth,
timeout = 100,
use_ssl = True,
verify_certs = True,
connection_class = RequestsHttpConnection,
embedding_function=bedrock_embeddings)

app = Flask(__name__)
CORS(app, supports_credentials=True)  # Enable CORS for all origins and allow credentials (cookies, HTTP authentication)

@app.route('/generate_responses', methods=['POST'])
def generate_images():
    

    data = request.json
    query = data.get('question')
    
    #Set up the message history for the appropriate user session. In this case I am setting it to session_id = 1 , kind of representing a user with unique id = ``
    message_history = DynamoDBChatMessageHistory(table_name="SessionTable", session_id="1")
    memory_chain = ConversationBufferMemory(memory_key="chat_history", chat_memory=message_history, return_messages=True)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=docsearch.as_retriever(), 
        memory=memory_chain,
        get_chat_history=_get_chat_history,
        # verbose=True,
        condense_question_prompt=condense_prompt_claude, 
        chain_type='stuff', # 'refine',
        #max_tokens_limit=300
    )

    # the LLMChain prompt to get the answer. the ConversationalRetrievalChange does not expose this parameter in the constructor
    qa.combine_docs_chain.llm_chain.prompt = PromptTemplate.from_template("""
    {context}

    Human: Use at maximum 3 sentences to answer the question inside the <q></q> XML tags. 

    <q>{question}</q>

    Do not use any XML tags in the answer. If the answer is not in the context say "Sorry, I don't know as the answer was not found in the context"

    Assistant:""")

    def extract_text(answer):
        # Regular expression to split text into text inside and outside XML-like tags
        split_pattern = r'(<\w+>[\s\S]*?<\/\w+>)|([^<>]+)'

        # Retrieve text inside and outside XML-like tags
        matches = re.findall(split_pattern, answer)

        # Extract text inside and outside XML-like tags
        text_inside_tags = ''.join(match[0] for match in matches if match[0])
        text_outside_tags = ''.join(match[1] for match in matches if match[1])

        # Determine which part to send based on content availability
        if text_outside_tags.strip():
            content_to_send = text_outside_tags.strip()
        else:
            content_to_send = text_inside_tags.strip()

        return content_to_send

    result = qa({"question": query})
    print(result['answer'])
    print("\n\n")
    answer = re.sub(r'<[^>]*>', '', extract_text(result['answer']))
    print(answer)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)