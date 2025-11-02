import os
import time
import logging
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GEN_API_KEY")

def _extract_text_from_obj(obj):
    try: return obj.choices[0].message.content
    except Exception: pass
    try: return obj['choices'][0]['message']['content']
    except Exception: pass
    try: return obj.choices[0].text
    except Exception: pass
    try:
        if hasattr(obj, "text"):
            return obj.text
        if hasattr(obj, "candidates"):
            c = getattr(obj, "candidates")
            if c and len(c) > 0:
                cand = c[0]
                if hasattr(cand, "content"):
                    return cand.content
                if isinstance(cand, dict) and "content" in cand:
                    return cand["content"]
    except Exception:
        pass
    try:
        if isinstance(obj, dict):
            for path in (("choices", 0, "message", "content"), ("choices", 0, "text"), ("text",)):
                cur = obj; ok = True
                for p in path:
                    try: cur = cur[p]
                    except Exception: ok = False; break
                if ok and isinstance(cur, str):
                    return cur
    except Exception:
        pass
    return None

def _try_openai(prompt, model="gpt-3.5-turbo", max_retries=3):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing")
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(f"openai client import failed: {e}")

    client = OpenAI(api_key=OPENAI_API_KEY)
    backoff = 0.5
    for attempt in range(1, max_retries+1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":prompt}]
            )
            text = _extract_text_from_obj(resp)
            if text:
                return text
            raise RuntimeError("OpenAI returned no text")
        except Exception as e:
            msg = str(e)
            logger.debug("OpenAI attempt %d failed: %s", attempt, msg)
            if "insufficient_quota" in msg.lower() or "quota" in msg.lower() or "insufficient" in msg.lower():
                raise RuntimeError(f"OpenAI quota/error: {msg}")
            if "429" in msg or "Too Many Requests" in msg:
                if attempt == max_retries:
                    raise RuntimeError(f"OpenAI retries exhausted: {msg}")
                time.sleep(backoff)
                backoff *= 2
                continue
            raise RuntimeError(f"OpenAI error: {msg}")
    raise RuntimeError("OpenAI unreachable")

def _try_gemini(prompt, candidate_models=None):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY / GOOGLE_GEN_API_KEY missing")
    try:
        import google.generativeai as genai
    except Exception as e:
        raise RuntimeError(f"google.generativeai import failed: {e}")
    genai.configure(api_key=GEMINI_API_KEY)

    # Prefer models observed on your account (from check_genai_models output)
    if candidate_models is None:
        candidate_models = [
            "models/gemini-flash-latest",
            "models/gemini-pro-latest",
            "models/gemini-2.5-flash",
            "models/gemini-2.5-pro",
            "models/gemini-2.5-flash-lite"
        ]

    # If genai supports listing, prioritize available models
    try:
        if hasattr(genai, "list_models"):
            avail = genai.list_models()
            names = []
            for m in avail:
                try:
                    names.append(m.name if hasattr(m, "name") else (m.get("name") if isinstance(m, dict) else None))
                except Exception:
                    continue
            found = [n for n in candidate_models if n in (names or [])]
            if found:
                candidate_models = found + [m for m in candidate_models if m not in found]
    except Exception:
        pass

    last_err = None
    for m in candidate_models:
        try:
            # Try GenerativeModel.generate_content (recommended)
            try:
                model_obj = genai.GenerativeModel(m)
                out = model_obj.generate_content(prompt)
                text = _extract_text_from_obj(out)
                if text:
                    return text
            except Exception as e1:
                # Fallback: try convenience functions if present
                try:
                    if hasattr(genai, "generate_text"):
                        out = genai.generate_text(model=m, prompt=prompt)
                        text = _extract_text_from_obj(out)
                        if text:
                            return text
                except Exception as e2:
                    last_err = f"{m}: {e1} | {e2}"
                    continue
        except Exception as e:
            last_err = f"{m}: {e}"
            continue
    raise RuntimeError(f"Gemini attempts failed. Last error: {last_err}")

def get_ai_response(prompt, model="gpt-3.5-turbo"):
    errors = []
    try:
        return _try_openai(prompt, model=model)
    except Exception as e:
        errors.append(f"OpenAI: {e}")

    try:
        return _try_gemini(prompt)
    except Exception as e:
        errors.append(f"Gemini: {e}")

    guidance = []
    if any("quota" in str(e).lower() or "insufficient_quota" in str(e).lower() for e in errors):
        guidance.append("OpenAI quota exhausted — rotate/upgrade key or check billing.")
    if any("missing" in str(e).lower() and "key" in str(e).lower() for e in errors):
        guidance.append("Missing API keys — set OPENAI_API_KEY and GOOGLE_GEN_API_KEY/GEMINI_API_KEY in .env")
    if not guidance:
        guidance.append("Check provider availability, keys, and client library versions.")

    raise RuntimeError("All providers failed. Details: " + " | ".join(errors) + " Guidance: " + " ".join(guidance))