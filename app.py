import os
from pathlib import Path
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
import logging
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv()

SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN')
OPENAI_API_TOKEN = os.getenv('OPENAI_API_TOKEN')

logging.basicConfig(level=logging.INFO)

# Here we load in the data from the text files created above.
ps = list(Path("docs/").glob("**/*.txt"))

data = []
for p in ps:
    with open(p) as f:
        data.append(f.read())

# Here we split the documents, as needed, into smaller chunks.
# We do this due to the context limits of the LLMs.


# Initialize the embedding model
embeddings = OpenAIEmbeddings()

# List to store the texts and their metadata
docs = []
metadatas = []

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = []

# Process each text file in the directory
for text in data:
    # Append the text and its metadata to the lists
    texts.extend(text_splitter.split_text(text))

# Generate embeddings and store (only need to run once)
embeddings = OpenAIEmbeddings()
vectors = embeddings.embed_documents(texts)
db = Chroma.from_texts(texts, embeddings)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})

from langchain.prompts import PromptTemplate
prompt_template = """Assistant is a large language model trained by OpenAI.
Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist\.
{context}
Question: {question}:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, max_tokens=300), chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)

chat_history_qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.4, max_tokens=300), retriever, combine_docs_chain_kwargs=chain_type_kwargs)
# Initializes your app with your bot token and socket mode handler
app = App(token=SLACK_BOT_TOKEN)

@app.message(".*")
def message_handler(message, say, client):
    logging.info("Received message: %s", message)
    handle_slack_message(message, message.get("thread_ts"), say, client)

@app.event("app_mention")
def handle_app_mentions(body, say, client):
    logging.info("Received app mention: %s", body)
    message = body['event']
    handle_slack_message(message, message.get("thread_ts"), say, client)


def handle_slack_message(message, thread_ts, say, client):
    logging.info(message)
    print("thread_ts -> ", thread_ts)
    if thread_ts:
        # Retrieve the thread's messages
        result = client.conversations_replies(channel=message['channel'], ts=thread_ts)

        chat_history = []
        current_human_messages = []

        for reply in result['messages'][:-1]:
            if 'bot_id' in reply:  # bot's message
                # Combine all preceding human messages into one
                human_message = "\n".join(current_human_messages)
                bot_message = reply['text']
                chat_history.append((human_message, bot_message))
                current_human_messages = []  # reset the list of human messages
            else:  # user's message
                current_human_messages.append(f"{reply['user']}: {reply['text']}")

        # If there are any remaining human messages after the last bot message, add them to chat history
        if current_human_messages:
            human_message = "\n".join(current_human_messages)
            chat_history.append((human_message, ""))
        for chat in chat_history:
            print(chat)
        output = chat_history_qa({"question": message['text'], "chat_history": chat_history})
        print(output)
        say(output["answer"], thread_ts=thread_ts)
    else:
        output = qa({"query": message['text']})
        print(output)
        say(output["result"])


# Start your app
if __name__ == "__main__":
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
