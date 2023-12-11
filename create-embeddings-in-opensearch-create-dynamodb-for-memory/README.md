docker build -t create-embeddings .


docker run -e -e AWS_ACCESS_KEY_ID=ABCD -e AWS_SECRET_ACCESS_KEY=1234567890ABCD  create-embeddings