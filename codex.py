import langchain, sys
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader, DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.cache import InMemoryCache
from langchain.storage import LocalFileStore
from timer import *
from colorama import Fore, Style

if (len(sys.argv) < 2):
    print("You would need to ask some question!")
    exit()

user_question = sys.argv[1]

#0. Load env
load_dotenv()
langchain.llm_cache = InMemoryCache()
langchain.verbose = False

# 1. Load
loader = DirectoryLoader("./data", "*.txt")
data, time = timeit(loader.load)
print("Finished loading data\t... " + str(time) + " seconds!")

# 2. Split
# chunk_size = 500 can produce error "Too many inputs. The max number of inputs is 16"
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
all_splits, time = timeit(text_splitter.split_documents,data)
print("Finished splitting data\t... " + str(time) + " seconds!")

#DONL cache embedding
underlying_embeddings = OpenAIEmbeddings()
fs = LocalFileStore("./cache/")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, fs, namespace=underlying_embeddings.model
)

# 3. Store
vectorstore, time = timeit(Chroma.from_documents, documents=all_splits, embedding=OpenAIEmbeddings())
print(Fore.YELLOW+"Finish creating index\t... " + str(time) + " seconds!") 

# cached version of vector store, cache is stored in /cache folder
vectorstore, time = timeit(Chroma.from_documents, documents=all_splits, embedding=cached_embedder)
print(Fore.YELLOW+"Finish creating index\t... (cached) " + str(time) + " seconds!")
print(Style.RESET_ALL)

# 4. Retrieve (Not very useful section)
# docs = vectorstore.similarity_search(user_question, k=1)

# 5. Generate
llm = AzureOpenAI(
    deployment_name="GPT-Test",
    model_name="gpt-35-turbo",
)
qa_chain, time = timeit(RetrievalQA.from_chain_type,llm,retriever=vectorstore.as_retriever())
print("Time for QA chain: "+str(time))
print("Raising question to LLM\t...")
response, time = timeit(qa_chain,{"query": user_question})

print("Query: "+response["query"])
print("Result: "+response["result"])
print(Fore.YELLOW+"Time for response: "+str(time))

# 6. Converse
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# print("Question: "+response["question"])
# print("Answer: "+response["answer"])
# print("Source URLs: "+response["sources"])