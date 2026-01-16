import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# 1. 페이지 설정
st.set_page_config(page_title="PDF 챗봇", page_icon="🤖")
st.title("📄 학교 공지사항 챗봇")

# API 키 설정
if "GEMINI_API_KEY" not in st.secrets:
    st.error("⚠️ Streamlit Secrets에 'GEMINI_API_KEY'를 설정해주세요!")
    st.stop()

os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

# 2. PDF 로드 및 학습 (상태 메시지 제거)
@st.cache_resource
def load_pdf_and_make_bot():
    file_path = "test.pdf"
    
    if not os.path.exists(file_path):
        return None
    
    try:
        # 조용히 로드 및 분할
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        
        # 임베딩 생성
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            task_type="retrieval_document"
        )
        
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        return vectorstore.as_retriever()
        
    except Exception:
        return None

# 데이터를 불러오는 동안 화면에 아무것도 띄우지 않거나 아주 짧게 대기
retriever = load_pdf_and_make_bot()

if retriever is None:
    st.error("❌ PDF를 불러올 수 없습니다. 'test.pdf' 파일을 확인해주세요.")
    st.stop()

# 3. 채팅 인터페이스
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 질문 입력
if prompt := st.chat_input("공지사항에 대해 궁금한 점을 물어보세요!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        try:
            with st.spinner("생각 중..."): # 최소한의 로딩 표시
                docs = retriever.invoke(prompt)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash", # 또는 "gemini-2.0-flash"
                    temperature=0
                )
                
                full_prompt = f"""다음 문서의 내용을 바탕으로 질문에 답변해주세요. 
문서에 관련 내용이 없다면 '죄송합니다. 학교 공지에 없는 내용입니다.'라고 답변해주세요.

문서 내용:
{context}

질문: {prompt}

답변:"""
                
                response = llm.invoke(full_prompt).content
                st.markdown(response)
                
        except Exception as e:
            response = "❌ 답변을 생성하는 중에 문제가 발생했습니다."
            st.error(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
