import os
from dotenv import load_dotenv
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
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

store = {}

def get_session_history(session_ids):
    if session_ids not in store: 
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  


llm_4o = ChatOpenAI(model="gpt-4o", temperature=0)

# model_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are an expert who answers the query based on the retrieved data."
#             "Answer in Korean.",
#         ),
#         # 대화기록용 key 인 chat_history 는 가급적 변경 없이 사용하세요!
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "#Question:\n{question} #Retrieved data: \n{retrieved_data}"), 
#     ]
# )

model_prompt = PromptTemplate(
    template="""
        You are an expert who answers the query based on the retrieved data.
        Answer in Korean.
        <Chat history>: {chat_history}
        <Question>: {query}
        <Retrieved data>: {retrieved_data}
    """,
    input_variables=["query", "retrieved_data", "chat_history"],
)

def chatbot(input):
    references = retriever(input)
    chain = model_prompt | llm_4o
    
    rag_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="query",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )
    
    answer = rag_with_history.invoke({"query": input, "retrieved_data": references}, config={"configurable": {"session_id": "tmp"}},)
    return answer.content

################## Chainlit ######################

@cl.on_message
async def run_convo(message: cl.Message):
    answer = chatbot(message.content)
    await cl.Message(content=answer).send()