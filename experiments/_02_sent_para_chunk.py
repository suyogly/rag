from corpus.loader import load_document
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs = load_document()

splitter = RecursiveCharacterTextSplitter(
    separators= ["\n\n", "\n"],
    chunk_size=500,
    chunk_overlap=0
)

content = splitter.split_documents(docs)

for idx, chunks in enumerate(content):
    print(f"id: {idx+1} \n chunk: {chunks.page_content} \n -- \n")