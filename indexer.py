from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores import FAISS

# 解析PDF，切成chunk片段
pdf_loader=PyPDFLoader('LLM.pdf',extract_images=True)   # 使用OCR解析pdf中图片里面的文字
chunks=pdf_loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=10))
'''
chunks 的内容：
[
    Document(page_content='人工智能 系列报告\n2024 年1月24日\n中航证券研究所 发布证券研究报告 请务必阅读正文后的免责条款部分\n行业评级：增持大模型专题报告：百模渐欲迷人眼， AI应用繁花开\n中航证券社会服务团队', metadata={'source': 'LLM.pdf', 'page': 0}),
    Document(page_content='分析师： 裴伊凡\n证券执业证书号： S0640516120002\n邮箱： peiyf@avicsec.com', metadata={'source': 'LLM.pdf', 'page': 0}), Document(page_content='2 资料来源：中 航证券研究所整理\uf070大模型演进：工业革命级的生产力工具。 目前， ANI已经广泛应用， AGI处于研发阶段，大模型是实现 AGI的重要路径。', metadata={'source': 'LLM.pdf', 'page': 1})
]
'''

# 加载embedding模型，用于将chunk向量化
embeddings=ModelScopeEmbeddings(model_id='iic/nlp_corom_sentence-embedding_chinese-base') 

# 将chunk插入到faiss本地向量数据库
vector_db=FAISS.from_documents(chunks,embeddings)
vector_db.save_local('LLM.faiss')

print('faiss saved!')
