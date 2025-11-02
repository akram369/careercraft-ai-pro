import os
from dotenv import load_dotenv
load_dotenv()
try:
    import google.generativeai as genai
except Exception as e:
    raise SystemExit(f"genai import failed: {e}")
KEY = os.getenv("GOOGLE_GEN_API_KEY") or os.getenv("GEMINI_API_KEY")
if not KEY:
    raise SystemExit("Missing GOOGLE_GEN_API_KEY / GEMINI_API_KEY in .env")
genai.configure(api_key=KEY)
try:
    models = genai.list_models()
except Exception as e:
    raise SystemExit(f"list_models failed: {e}")
for m in models:
    try:
        print(m.name if hasattr(m, "name") else (m.get("name") if isinstance(m, dict) else str(m)))
    except Exception:
        print(str(m))