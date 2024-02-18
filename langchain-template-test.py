from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import RetrievalQA


loader = CSVLoader("./data/kdrama.csv")
documents = loader.load()

# 데이터를 불러와서 텍스트를 일정한 수로 나누고 구분자로 연결하는 작업
text_splitter = CharacterTextSplitter(
	chunk_size=1000, 
    chunk_overlap=0, 
    separator="\n"
    )

texts = text_splitter.split_documents(documents)

# intfloat/multilingual-e5-large Embedding 모델은 94개국어의 텍스트를 임베딩하는 모델

# 임베딩 모델 로드
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

index = FAISS.from_documents(
	documents=texts,
	embedding=embeddings,
	)

# faiss_db 로 로컬에 저장하기
index.save_local("faiss_db")

# faiss_db 로 로컬에 로드하기
docsearch = FAISS.load_local("faiss_db", embeddings)

# Langchain, RetrievalQA 로 질의하기
# VectorDB 로 Embedding 한 Llama2 모델에게 한국 드라마 질문하기

# 유사도 0.7로 임베딩 필터를 저장
# 유사도에 맞추어 대상이 되는 텍스트를 임베딩함
embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings, 
    similarity_threshold=0.70
)

compression_retriever = ContextualCompressionRetriever(
	# embeddings_filter 설정
    base_compressor=embeddings_filter, 
    # retriever 를 호출하여 검색쿼리와 유사한 텍스트를 찾음
    base_retriever=docsearch.as_retriever()
)

# RetrievalQA 클래스의 from_chain_type이라는 클래스 메서드를 호출하여 질의응답 객체를 생성
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=compression_retriever)

response = qa.run("Among Korean dramas, please recommend 1 medical dramas about hospital life.")