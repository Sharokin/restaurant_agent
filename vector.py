from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

CSV_PATH = "data/restaurant.csv"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "restaurant_reviews"

df = pd.read_csv(CSV_PATH)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
add_documents = not os.path.exists(DB_PATH)

# prep docs
if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content=f"{row['Title']} {row['Review']}",
            metadata={
                "rating": row["Rating"],
                "date": row["Date"]
            },
            id=str(i)
        )
        documents.append(document)
        ids.append(str(i))

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5} # default
)
