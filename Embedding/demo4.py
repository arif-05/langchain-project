# similarity checking

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

# two sentences
s1 = "I love programming in Python."
s2 = "Python coding is my passion."

# one different sentence
s3 = "The weather is nice today."
emb1 = model.encode(s1)
emb2 = model.encode(s2) 
emb3 = model.encode(s3)


# cosine similarity

print(util.cos_sim(emb1, emb2)) # similar sentences
print(util.cos_sim(emb1, emb3)) # different sentences
