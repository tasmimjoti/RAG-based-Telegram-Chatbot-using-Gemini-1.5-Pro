import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import logging
import time

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

TELEGRAM_TOKEN = os.getenv("<TELEGRAM TOKEN>")
GOOGLE_API_KEY = os.getenv("<GOOGLE API KEY>")
PERSIST_DIR = './db/gemini/'  # Replace with your actual directory

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]="<LANGCHAIN API KEY>"
os.environ["LANGCHAIN_PROJECT"]="<YOUR LANGCHAIN PROJECT NAME>"

# Initialize chat history
history = []

# Initialize the Gemini Pro 1.5 model
model = ChatGoogleGenerativeAI(
    model="gemini-pro", 
    temperature=0.1, 
    convert_system_message_to_human=True
)

# Configure Google Generative AI with the API key
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

if not os.path.exists(PERSIST_DIR):
    # Data Pre-processing
    pdf_loader = DirectoryLoader("data/", glob="./*.pdf", loader_cls=PyPDFLoader)
    
    try:
        pdf_documents = pdf_loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
        pdf_context = "\n\n".join(str(p.page_content) for p in pdf_documents)
        pdfs = splitter.split_text(pdf_context)
        vectordb = Chroma.from_texts(pdfs, embeddings, persist_directory=PERSIST_DIR)
        vectordb.persist()
    except Exception as e:
        logger.error(f"Error loading and processing PDF documents: {e}")
        raise
else:
    try:
        vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    except Exception as e:
        logger.error(f"Error loading persisted vector database: {e}")
        raise

# Initialize retriever and query chain
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
query_chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Hello! I am a chatbot. How can I assist you?')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    history.append({'role': 'user', 'content': user_message})

    prompt = (
        "YOUR CUSTOMIZED PROMPT"
    )
    
    # Formulate the complete query
    query = f"{prompt}\n\nUser Question: {user_message}"
    
    # Get the response from the query chain
    response = query_chain({"query": query})
    bot_response = response['result']

    history.append({'role': 'assistant', 'content': bot_response})
    
    # Check for empty response
    if not bot_response:
        bot_response = "I'm sorry, I couldn't find an answer to your question."

    # Split long messages
    if len(bot_response) > 4096:
        for i in range(0, len(bot_response), 4096):
            await update.message.reply_text(bot_response[i:i+4096])
    else:
        await update.message.reply_text(bot_response)

def main() -> None:
    import telegram
    retry_attempts = 5
    for attempt in range(retry_attempts):
        try:
            app = ApplicationBuilder().token('<TELEGRAM TOKEN>').build()
            
            app.add_handler(CommandHandler("start", start))
            app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
            
            logger.info("Bot is starting...")
            app.run_polling()
            break
        except telegram.error.NetworkError as e:
            logger.error(f"Network error: {e}. Attempt {attempt + 1} of {retry_attempts}")
            time.sleep(5)  # wait for 5 seconds before retrying
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise

if __name__ == '__main__':
    main()
