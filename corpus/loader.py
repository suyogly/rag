from langchain_community.document_loaders import UnstructuredMarkdownLoader

md_file = "corpus/data/nepal_advanced.md"

def load_document():
    loader = UnstructuredMarkdownLoader(
        md_file,
        mode="single",
        strategy="fast"
    )

    content = loader.load()
    return content

# print(load_document())