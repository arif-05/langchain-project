from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embeddings = HuggingFaceEmbeddings(
    model = "sentence-transformers/all-MiniLM-L6-v2"
)


documents = [
    "today is a sunny day", 
    "I love programming in Python.",
    "Python coding is my passion.",
    "The weather is nice today."
    ]

query = "I enjoy coding in Python."

doc_vector = embeddings.embed_documents(documents)
query_vector = embeddings.embed_query(query)    

scores = cosine_similarity([query_vector], doc_vector)

print(scores) 

index = np.argmax(scores)
score = scores[0][index]

print("Query:", query)
print("Document:", documents[index])
print("Score:", score)