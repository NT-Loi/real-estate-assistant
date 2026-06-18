from google import genai
from dotenv import load_dotenv
load_dotenv()
import os
PROJECT_ID = os.environ.get("PROJECT_ID", "")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location="global",
)

for chunk in client.models.generate_content_stream(
    model=MODEL,
    contents="explain about kv cache",
):
    if chunk.text:
        print(chunk.text, end="", flush=True)

print()