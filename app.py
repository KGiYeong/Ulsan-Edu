import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 1. 페이지 설정
st.set_page_config(page_title="PDF 테스트 챗봇", page_icon="🤖")
st.title("📄 GitHub 파일 읽기 테스트")

# API 키 설정 (Streamlit Secrets 필수)
if "GEMINI_API_KEY" not in st.secrets:
    st.error("설정(Secrets)에 'GEMINI_API_KEY'를 넣어주세요.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

# 2. PDF 로드 및 학습 (test.pdf 전용)
@st.cache_resource
def load_pdf_and_make_bot():
    file_path = "test.pdf" # GitHub에 올린 파일 이름
    
    if not os.path.exists(file_path):
        st.error(f"'{file_path}' 파일을 찾을 수 없습니다. GitHub에 파일을 올렸는지 확인해주세요.")
        return None

    # PDF 읽기
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # 텍스트 나누기
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # 임베딩(공부하기) 및 저장소 만들기
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()

# 챗봇 준비
retriever = load_pdf_and_make_bot()

# 3. 채팅 인터페이스
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("test.pdf 내용에 대해 물어보세요!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if retriever is None:
            response = "파일이 없어 답변을 드릴 수 없습니다."
        else:
            # 최신 gemini-2.5-flash 모델 호출
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
            
            # 답변 생성 (문서에 없으면 모른다고 하기)
            with st.spinner("답변을 찾는 중..."):
                response = qa_chain.run(f"{prompt} (문서에 내용이 없으면 '죄송합니다. 학교 공지에 없는 내용입니다.'라고 답해줘)")
                st.markdown(response)
                
    st.session_state.messages.append({"role": "assistant", "content": response})
