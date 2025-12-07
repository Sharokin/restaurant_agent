from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vector import retriever

app = FastAPI()

# Allow your frontend to access the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or put your frontend URL if you want to be strict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    question = req.question
    reviews = retriever.invoke(question)
    from langchain_ollama.llms import OllamaLLM
    from langchain_core.prompts import ChatPromptTemplate

    model = OllamaLLM(model ="llama3.2")
    template = """
    You are a food expert that knows how to anser questions about restaurant reviews
    Here are some reviews of a restaurant: {reviews}
    This is the question to answer: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    result = chain.invoke({"reviews": reviews, "question": question})
    return {"answer": result}
