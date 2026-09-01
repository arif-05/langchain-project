from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model = "gemini-embedding-2"
)

vector = embeddings.embed_documents([
    "today is a sunny day",
    "today is a rainy day",
    "today is a snowy day",
    "today is a cloudy day"
])

print(len(vector), len(vector[1]))
