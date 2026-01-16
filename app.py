import streamlit as st

import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS



# 1. 페이지 설정

st.set_page_config(page_title="PDF 테스트 챗봇", page_icon="🤖")

st.title("📄 GitHub 파일 읽기 테스트")



# API 키 설정

if "GEMINI_API_KEY" not in st.secrets:

    st.error("⚠️ Streamlit Secrets에 'GEMINI_API_KEY'를 설정해주세요!")

    st.stop()



os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]



# 2. PDF 로드 및 학습

@st.cache_resource

def load_pdf_and_make_bot():

    file_path = "test.pdf"

    

    # 파일 존재 확인

    if not os.path.exists(file_path):

        st.error(f"❌ '{file_path}' 파일을 찾을 수 없습니다.")

        st.info("GitHub 저장소에 test.pdf 파일이 있는지 확인해주세요.")

        return None

    

    try:

        st.info("📄 PDF 파일을 읽는 중...")

        loader = PyPDFLoader(file_path)

        docs = loader.load()

        st.success(f"✅ PDF 로드 완료: {len(docs)}페이지")

        

        st.info("✂️ 텍스트를 나누는 중...")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        splits = text_splitter.split_documents(docs)

        st.success(f"✅ 텍스트 분할 완료: {len(splits)}개 조각")

        

        st.info("🧠 임베딩 생성 중... (시간이 걸릴 수 있습니다)")

        # 임베딩 모델 - models/text-embedding-004 사용

        embeddings = GoogleGenerativeAIEmbeddings(

            model="models/text-embedding-004",

            task_type="retrieval_document"

        )

        

        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

        st.success("✅ 학습 완료!")

        

        return vectorstore.as_retriever()

        

    except Exception as e:

        st.error(f"❌ 오류 발생: {str(e)}")

        st.info("💡 Gemini API 할당량을 확인하거나, 잠시 후 다시 시도해주세요.")

        return None



retriever = load_pdf_and_make_bot()



if retriever is None:

    st.warning("⚠️ 챗봇을 초기화할 수 없습니다. 위의 오류를 확인해주세요.")

    st.stop()



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

        try:

            with st.spinner("답변을 찾는 중..."):

                docs = retriever.invoke(prompt)

                context = "\n\n".join([doc.page_content for doc in docs])

                

                llm = ChatGoogleGenerativeAI(

                    model="gemini-2.5-flash",

                    temperature=0

                )

                

                full_prompt = f"""다음 문서의 내용을 바탕으로 질문에 답변해주세요. 

문서에 관련 내용이 없다면 '죄송합니다. 학교 공지에 없는 내용입니다.'라고 답변해주세요.
