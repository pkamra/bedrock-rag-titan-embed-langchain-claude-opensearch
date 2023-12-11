# Botkit Starter Kit

This is a Botkit starter kit for web, created with the [Yeoman generator](https://github.com/howdyai/botkit/tree/master/packages/generator-botkit#readme).

To complete the configuration of this bot, make sure to update the included `.env` file with your platform tokens and credentials.

[Botkit Docs](https://github.com/howdyai/botkit/blob/main/packages/docs/index.md)

This bot is powered by [a folder full of modules](https://github.com/howdyai/botkit/blob/main/packages/docs/core.md#organize-your-bot-code). 
Edit the samples, and add your own in the [features/](features/) folder.


export DOCKERHOST=$(ifconfig | grep -E "([0-9]{1,3}\.){3}[0-9]{1,3}" | grep -v 127.0.0.1 | awk '{ print $2 }' | cut -f2 -d: | head -n1)


docker build -t  frontend .

docker run -p 3000:3000 -e QUEST_URL=http://${DOCKERHOST}:8000/generate_responses  frontend
