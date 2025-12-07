from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

app = FastAPI()

model = OllamaLLM(model="llama3.2")

template = """
You are a food expert that knows how to anser questions about restaurant reviews

Here are some reviews of a restaurant: {reviews}

This is the question to answer: {question}

"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(request: ChatRequest):
    docs = retriever.invoke(request.question)
    
    reviews_text = "\n\n".join([doc.page_content for doc in docs])

    result = chain.invoke({
        "reviews": reviews_text,
        "question": request.question
    })

    return {"answer": result}