"""Phase 0 smoke: one Gemini call logged as a Langfuse trace."""

import os

from dotenv import load_dotenv
from google import genai
from langfuse import Langfuse

load_dotenv()

MODEL = "gemini-3.5-flash-lite"
PROMPT = "Réponds uniquement par le mot: OK"

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)

with langfuse.start_as_current_observation(
    name="smoke",
    as_type="generation",
    model=MODEL,
    input=PROMPT,
):
    response = client.models.generate_content(model=MODEL, contents=PROMPT)
    langfuse.update_current_generation(output=response.text)
    trace_url = langfuse.get_trace_url()

langfuse.flush()

print(f"Gemini replied: {response.text.strip()}")
print(f"Trace: {trace_url}")
