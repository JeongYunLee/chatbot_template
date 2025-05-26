import os
from dotenv import load_dotenv
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import chainlit as cl

load_dotenv()

################# Retriever ######################

client = chromadb.PersistentClient('chroma/')
embedding = OpenAIEmbeddings(model='text-embedding-3-large')  
vectorstore = Chroma(client=client, collection_name="dataset", embedding_function=embedding)

def retriever(input):
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(input)
    reference = [docs[i].page_content for i in range(len(docs))]
    return reference

################# LLM ######################

llm_4o = ChatOpenAI(model="gpt-4o", temperature=0)

model_prompt = PromptTemplate(
    template="""
        You are an expert who answers the query based on the retrieved data.
        Answer in Korean.
        <Question>: {query}
        <Retrieved data>: {retrieved_data}
    """,
    input_variables=["query", "retrieved_data"],
)

def chatbot(input):
    references = retriever(input)
    chain = model_prompt | llm_4o 
    answer = chain.invoke({"query": input, "retrieved_data": references})
    return answer.content

################## Chainlit ######################

@cl.on_message
async def run_convo(message: cl.Message):
    answer = chatbot(message.content)
    await cl.Message(content=answer).send()
    
