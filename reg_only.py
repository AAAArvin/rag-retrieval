from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores import FAISS
import os

os.environ['VLLM_USE_MODELSCOPE'] = 'True'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 加载embedding模型，用于将query向量化
embeddings = ModelScopeEmbeddings(model_id='iic/nlp_corom_sentence-embedding_chinese-base')

# 加载faiss向量库，用于知识召回
vector_db = FAISS.load_local('LLM.faiss', embeddings, allow_dangerous_deserialization=True)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})
print('-------------------')
print(retriever.get_relevant_documents("增持大模型专题报告"))
