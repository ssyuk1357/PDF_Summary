# pdf-reader

# import

import os
from dotenv import load_dotenv
import openai
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
# 문맥 파악
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
# 사전 학습 모델 불러오기
from langchain import hub

# api 키 불러오기
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key

# pdf 불러오기
# pdf 갯수에 맞춰 수정
pdf1 = "the pdf1 you want"
pdf2 = "the pdf2 you want"
pdf3 = 'the pdf3 you want'

# pdf 갯수에 맞춰 수정
loaders = [PyPDFLoader(pdf1), PyPDFLoader(pdf2), PyPDFLoader(pdf3)]

docs = []

# 문서 타입 확인
for i, loader in enumerate(loaders):
  pages = loader.load()
  docs.extend(pages)

# 문서 스플릿
text_splitter = CharacterTextSplitter(
    # 정밀한 결과를 위해서 계속해서 바꿔야 하는 수치
    # seprator은 스플릿 기준 설정
    chunk_size=1500, chunk_overlap=150, separator='. ')
chunks = text_splitter.split_documents(docs)

# 벡터 변환
embeddings = OpenAIEmbeddings()

vectordb = Chroma.from_documents(documents=chunks,embedding=embeddings)
print(vectordb._collection.count())


llm = ChatOpenAI(model_name='gpt-3.5-turbo')
retriever = vectordb.as_retriever()
prompt = ChatPromptTemplate.from_messages(
    [
        # 한글에 비해 영어가 토큰 수가 적다 -> 값이 싸진다
        ('system', '''You are a helpful assistant. Answer questions using only the fllowing context. If you don't know the answer just say you dont know, do not make it up " \n\n {context}'''),
        ('user','{context}'),
        ('human', '{question}')
    ]
)
# 3가지를 합쳐서 사용 하겠다.
chain = ({'context':retriever, 'question':RunnablePassthrough()} | prompt | llm)
# 질문창
# 체인을 통해 invoke에 전달
chain.invoke("난장이가 쏘아올린 작은 공의 결말은?")

# 질의 응답 챗봇
# 데이터 베이스 기반 응답시 retrievalQA 사용
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='map_rerank',
    retriever=retriever
    )

chain.run('운수좋은날 주인공')


# 타 모델과의 연동 - rag
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.1)
memory = ConversationBufferMemory(memory_key='chat_history')

retriver = vectordb.as_retriever()

rag_prompt = hub.pull('rlm/rag-prompt')

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    verbose=False
)

chain({'question':'소나기의 주인공은?'})