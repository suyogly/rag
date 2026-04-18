from corpus.loader import load_document
from langchain_text_splitters import RecursiveCharacterTextSplitter

doc = load_document()

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size = 500,
    chunk_overlap = 200
)

'''
Recursive Character Text Splitter is an
interesting splitter because, depending
upon our chunk size, it automatically adjusts
to use the fallback options.

For our 500 chunk size, it first checks if
chunking by paragraph break fits the paragraph
inside the 500 chunk boundary, if not, it chunks
that long paragraph and cuts from the sentence.. 
and so on.

'''

chunks = splitter.split_documents(doc)

for idx, chunk in enumerate(chunks):
    print(f"id: {idx+1} \n chunk: {chunk.page_content} \n -- \n")