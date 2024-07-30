import joblib
from langchain.chains import create_retrieval_chain, create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

def load_model(): 
  config=joblib.model('model_config.joblib')
  mistral_llm=HuggingFaceEndpoint(
    task='text-generattion',
    endpoint_url=mistral_url,
    max_new_tokens=512,
    temperature=0.7,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

  custom_rag_prompt=PromptTemplate.from_template(template)
  loader=WebBaseLoader(web_path=('https://www.airnewzealand.co.nz/checked-in-baggage'))
  docs=loader.load()
  text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
  splits=text_splitter.split_documents(docs)
  embedding_model=SentenceTransformerEmbeddings(model_name='all-mpnet-base-v2')
  vectorstore=Chroma.from_documents(documents=splits,embedding=embedding_model)
  retriever=vectorstore.as_retriever(search_type='similarity',search_kwargs={'k':6})

  rag_chain=create_retrieval_chain(retriever,create_stuff_documents_chain(mistral_llm, custom_rag_prompt))

  return rag_chain, retriever

def format_docs(docs):
  return '\n\n'.join(doc.page_content for doc in docs)

