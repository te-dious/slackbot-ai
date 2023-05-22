import os
from pathlib import Path
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import logging
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv()

SLACK_BOT_TOKEN = os.getenv('SLACK_CHATWOOT_BOT_TOKEN')
SLACK_APP_TOKEN = os.getenv('SLACK_CHATWOOT_APP_TOKEN')
OPENAI_API_TOKEN = os.getenv('OPENAI_API_TOKEN')

logging.basicConfig(level=logging.INFO)

# Here we load in the data from the text files created above.
ps = list(Path("docs/").glob("**/*_chatwoot.txt"))

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
db = Chroma.from_texts(texts, embeddings, persist_directory="chatwoot")

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})

from langchain.prompts import PromptTemplate
prompt_template = """Assistant is a large language model trained by OpenAI.
If question are relevant to context, AI answer using the context, and not make up any answer.
If you are asked to summarise the conversation, you summarise the whole chat history provided.
You are the customer success team manager for a Thai insurtech company that supports insurance agents in selling insurance policies and earning commissions.
Insurance agents can either chat with staff members for assistance during the buying process or use the online platform provided by the insurtech company.
The insurance purchasing process involves obtaining a quote, selecting a plan, submitting required documents, reporting a sale of a policy by giving customer details, making a payment, issuing a policy, and completing the transaction when the policy is physically delivered.
Agents can also request renewals for existing policies bought from the company before.
In the Thai market, some manual processes may be involved, such as sending payment proof, paying premiums in installments, requesting a physical car inspection, manually delivering the policy via postal service, endorsing an existing policy, or change of agent ( COA - switching from one broker to your company).
{context}
Question: {question}:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

llm = ChatOpenAI(model_name="gpt-4", temperature=0)
chat_history_qa = ConversationalRetrievalChain.from_llm(llm, retriever, combine_docs_chain_kwargs=chain_type_kwargs)

# Initializes your app with your bot token and socket mode handler
app = App(token=SLACK_BOT_TOKEN)

@app.message(".*")
def message_handler(message, say, client):
    logging.info("Received message: %s", message)
    handle_slack_message(message, message.get("ts"), say, client)

@app.event("app_mention")
def handle_app_mentions(body, say, client):
    logging.info("Received app mention: %s", body)
    message = body['event']
    handle_slack_message(message, message.get("ts"), say, client)


def handle_slack_message(message, thread_ts, say, client):
    logging.info(message)
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
        answer = f'{output["answer"]}'
        say(answer, thread_ts=thread_ts)

    except Exception as e:
        say("Please Try Again", thread_ts=thread_ts)



# Start your app
if __name__ == "__main__":
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
