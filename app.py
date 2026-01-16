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

# 2. PDF 로드 및 학습 (과정 메시지 숨김)
@st.cache_resource
def load_pdf_and_make_bot():
    file_path = "test.pdf"
    if not os.path.exists(file_path):
        return None
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        
        # 임베딩 모델 설정
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        return vectorstore.as_retriever()
    except Exception as e:
        st.error(f"데이터 준비 중 오류: {e}")
        return None

retriever = load_pdf_and_make_bot()

# 3. 채팅 인터페이스 (이전 대화 표시)
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. 질문 입력 및 답변 생성
if prompt := st.chat_input("test.pdf 내용에 대해 물어보세요!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        try:
            with st.spinner("답변을 찾는 중..."):
                docs = retriever.invoke(prompt)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # 안정적인 모델명 사용
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    temperature=0
                )
                
                # 프롬프트 구성 (줄바꿈 오류 방지를 위해 정돈)
                instruction = "다음 문서 내용을 바탕으로 질문에 친절하게 답변해주세요. 내용에 없다면 '학교 공지에 없는 내용입니다.'라고 하세요."
                full_prompt = f"{instruction}\n\n[문서 내용]\n{context}\n\n[질문]\n{prompt}"
                
                response = llm.invoke(full_prompt).content
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
        except Exception as e:
            st.error(f"❌ 답변 생성 중 오류가 발생했습니다: {e}")
