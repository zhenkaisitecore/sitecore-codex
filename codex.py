import langchain, sys
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader, DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from langchain.cache import InMemoryCache
from langchain.memory import ConversationBufferMemory

if (len(sys.argv) < 2):
    print("You would need to ask some question!")
    exit()

user_question = sys.argv[1]

#0. Load env
load_dotenv()
langchain.llm_cache = InMemoryCache()
langchain.verbose = True

# 1. Load
loader = DirectoryLoader("./data", "*.txt")
data = loader.load()
print("Finished loading data\t...")

# 2. Split
# chunk_size = 500 can produce error "Too many inputs. The max number of inputs is 16"
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
all_splits = text_splitter.split_documents(data)
print("Finished splitting data\t...")

# 3. Store
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
print("Finish creating index\t...")

# 4. Retrieve (Not very useful section)
# docs = vectorstore.similarity_search(user_question, k=1)

# 5. Generate
chat = AzureChatOpenAI(
    deployment_name="GPT-Test",
)
qa_chain = RetrievalQA.from_chain_type(chat,retriever=vectorstore.as_retriever())
print("Raising question to LLM\t...")
response = qa_chain({"query": user_question})

print("Query: "+response["query"])
print("Result: "+response["result"])

# 6. Converse
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# print("Question: "+response["question"])
# print("Answer: "+response["answer"])
# print("Source URLs: "+response["sources"])