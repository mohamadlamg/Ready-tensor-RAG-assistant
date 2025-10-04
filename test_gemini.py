import os 
from json import load
from dotenv import load_dotenv
load_dotenv()

gemini_api = os.getenv("GEMINI_API_KEY")


from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
response = llm.invoke("Give me a metaphor for vector embeddings")
print(response.content)