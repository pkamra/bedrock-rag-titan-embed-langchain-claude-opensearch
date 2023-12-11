docker build -t api-claude-questions-with-conversational-memory .


docker run -p 8000:5000 -e AWS_ACCESS_KEY_ID=ABCD -e AWS_SECRET_ACCESS_KEY=1234567890ABCD  -e AOSS_HOST=https://zl8jsxuuedvh2ioix474.us-east-1.aoss.amazonaws.com:443 api-claude-questions-with-conversational-memory