from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model = "sentence-transformers/all-MiniLM-L6-v2"
)


documents = [
    "today is a sunny day",
    "today is a rainy day", 
    "today is a snowy day"]

vector = embeddings.embed_documents(documents)

print(len(vector), len(vector[1]))