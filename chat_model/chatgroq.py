from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b"
)

result = llm.invoke("what is the capital of Bangladesh?")

print(result.content)
