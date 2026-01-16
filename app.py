import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 1. 페이지 설정
st.set_page_config(page_title="우리학교 알림이 챗봇", page_icon="🏫")
st.title("🏫 학교 소식 무엇이든 물어보세요!")
st.info("공지사항이나 가정통신문 PDF 파일들을 업로드하면 챗봇이 내용을 학습합니다.")

# API 키 설정
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Secrets에 'GEMINI_API_KEY'를 설정해주세요.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

# 2. 파일 업로드 및 벡터 DB 생성 함수
def create_vector_db(uploaded_files):
    # 임시 디렉토리에 파일 저장
    if not os.path.exists("temp_docs"):
        os.makedirs("temp_docs")
    
    all_documents = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join("temp_docs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # PDF 로드
        loader = PyPDFLoader(file_path)
        all_documents.extend(loader.load())

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(all_documents)

    # 벡터 저장소 생성 (gemini-2.5-flash와 호환되는 임베딩)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

# 사이드바에서 파일 업로드
with st.sidebar:
    st.header("파일 업로드")
    uploaded_files = st.file_uploader("PDF 파일을 선택하세요 (여러 개 가능)", type="pdf", accept_multiple_files=True)
    process_button = st.button("학습 시작")

if process_button and uploaded_files:
    with st.spinner("학교 소식을 읽고 있습니다... 잠시만 기다려주세요!"):
        st.session_state.vector_db = create_vector_db(uploaded_files)
        st.success("준비 완료! 이제 질문을 시작하세요.")

# 3. 채팅 인터페이스
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("예: 이번주 준비물이 뭐야?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if "vector_db" not in st.session_state:
            response = "먼저 왼쪽에서 학교 서류(PDF)를 업로드하고 '학습 시작'을 눌러주세요!"
        else:
            # RAG 체인 구성 (최신 모델 gemini-2.5-flash 사용)
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
            
            # 검색 및 답변
            retriever = st.session_state.vector_db.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            # 답변 생성 시 프롬프트 보강 (모르는 내용은 모른다고 하기)
            result = qa_chain({"query": f"{prompt} (만약 문서에 관련 내용이 없다면 '학교에서 안내된 바가 없습니다'라고 답해줘)"})
            response = result["result"]
            
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
