import os
import requests
import json
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
from langchain.llms import OpenAI
load_dotenv()

SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN')
OPENAI_API_TOKEN = os.getenv('OPENAI_API_TOKEN')
REDASH_API_TOKEN = os.getenv('REDASH_API_TOKEN')

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

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":5})

from langchain.prompts import PromptTemplate
prompt_template = """As a senior analyst, write a detailed and correct MySql query to answer the analytical question based on the context you have.
You should not assume anything, if you don't know the schema just don't make up any relation.
If user sends some error message, use that error message as feedback to improve the last query sent.
Your response should be only sql query. If any CTE is used please define that as well in the query itself.
{context}
Question: {question}: """
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, such that no context of historical chat is missed and the follow up question is answered accordingly.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

llm = ChatOpenAI(model_name="gpt-4", temperature=0)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)

chat_history_qa = ConversationalRetrievalChain.from_llm(llm, retriever, condense_question_prompt=CONDENSE_QUESTION_PROMPT, combine_docs_chain_kwargs=chain_type_kwargs)
# Initializes your app with your bot token and socket mode handler
app = App(token=SLACK_BOT_TOKEN)

def run_sql_query(mysql_query, text):
    """Creates a Redash query from a MySQL query string.

    Args:
        mysql_query: The MySQL query string.

    Returns:
        The Redash query object.
    """
    llm = OpenAI(model_name="gpt-4", temperature=0)
    name = llm(f'convert the following text in triple ticks into a short crisp title. the text is as follows ```{text}```')

    # Create the Redash query object.
    query = {
        "data_source_id": 1,
        "query": mysql_query,
        "name": name,
        "description": text,
        "tags": ["automated"],
    }
    

    # Post the query to Redash.
    response = requests.post(
        "https://redash.insbee.sg/api/queries",
        headers={"Authorization": f"Key {REDASH_API_TOKEN}"},
        data=json.dumps(query),
    )

    # Check the response status code.
    if response.status_code not in (200, 201):
        raise Exception("Failed to create Redash query: {}".format(response.status_code))

    # Return the Redash query object.
    res = response.json()
    return f"https://redash.insbee.sg/queries/{res['id']}"

@app.message(".*")
def message_handler(message, say, client):
    logging.info("Received message: %s", message)
    handle_slack_message(message, message.get("thread_ts"), say, client, True)

@app.event("app_mention")
def handle_app_mentions(body, say, client):
    logging.info("Received app mention: %s", body)
    message = body['event']
    handle_slack_message(message, message.get("thread_ts"), say, client)


def handle_slack_message(message, thread_ts, say, client, direct=False):
    logging.info(message)
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
        try:
            output = chat_history_qa({"question": message['text'], "chat_history": chat_history})
            say(output["answer"], thread_ts=thread_ts)
        except Exception as e:
            say("Please Try Agian", thread_ts=thread_ts)
    else:
        try:
            output = qa({"query": message['text']})
            try:
                logging.info(output)
                answer = f'{output["result"]}'
                say(answer)
                out = run_sql_query(output["result"], message['text'])
                logging.info(out)
                say(out)
            except Exception as e:
                say(e)

        except Exception as e:
            say("Please Try Again")


# Start your app
if __name__ == "__main__":
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
