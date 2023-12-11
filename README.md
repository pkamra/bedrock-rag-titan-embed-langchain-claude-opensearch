This code repository is for the medium article https://medium.com/@piyalikamra/build-a-conversational-car-savvy-digital-assistant-with-memory-using-anthropic-claude-on-bedrock-1f86bb0c89c9

Here are the high level Steps :-

#1)  The code for creating embeddings and initializing the OpenSearch Vector Database is in folder /create-embeddings-in-opensearch-create-dynamodb-for-memory . The instructions for building the Docker container are in the README of that folder.

#2) The code for the backend API to retrieve the information from OpenSearch and for maintaining conversational memory is in the folder /api-for-bedrock-anthropic-claude-llm-with-conversational-memory . The instructions for building the Docker container for the API are in the README of that folder.

#3) The front end code is in the folder /digital-assistant-frontend. The instructions for building the Docker container for the frontened are in the README of that folder.