from corpus.loader import load_document
from langchain_text_splitters import CharacterTextSplitter

docs = load_document()

'''
fixed size chunking with "" separator
and no chunk overlap
'''
fixed_splitter = CharacterTextSplitter(
    separator="",
    chunk_size = 100,
    chunk_overlap = 0 # by default it has overlap of 200
)

content = fixed_splitter.split_documents(docs)

for idx, chunks in enumerate(content):
    print(f"id: {idx+1} \n chunk: {chunks.page_content} \n -- \n")



'''
fixed size chunking with " " separator
and no chunk overlap
'''
fixed_splitter_space = CharacterTextSplitter(
    separator=" ",
    chunk_size = 100,
    chunk_overlap = 0 # by default it has overlap of 200
)

content2 = fixed_splitter_space.split_documents(docs)

for idx, chunks in enumerate(content2):
    print(f"id: {idx+1} \n chunk: {chunks.page_content} \n -- \n")



'''
fixed size chunking with " " separator
and chunk overlap of 50
'''
fixed_splitter_overlap = CharacterTextSplitter(
    separator=" ",
    chunk_size = 100,
    chunk_overlap = 50 # by default it has overlap of 200
)

content3 = fixed_splitter_overlap.split_documents(docs)

for idx, chunks in enumerate(content3):
    print(f"id: {idx+1} \n chunk: {chunks.page_content} \n -- \n")
