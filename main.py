from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriver

model = OllamaLLM(model ="llama3.2")

template = """
You are a food expert that knows how to anser questions about restaurant reviews

Here are some reviews of a restaurant: {reviews}

This is the question to answer: {question}

"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    question = input("Ask a question about the restaurant(q to quit): ")
    print("\n")
    if question == "q":
        break

    reviews = retriver.invoke(question)

    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)