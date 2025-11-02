# app.py
import os
import json
import random
import hashlib
import difflib
import re
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv
import requests
import streamlit as st

# PDF + DOC utilities
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PyPDF2 import PdfReader
from docx import Document

# ML / NLP
from sklearn.feature_extraction.text import TfidfVectorizer

# Project helpers
from youtube_fetcher import get_learning_resources
import requests

BACKEND_URL = "https://careercraft-ai-pro-backend.onrender.com/generate"

def get_ai_response(prompt, model="gpt-3.5-turbo"):
    try:
        res = requests.post(BACKEND_URL, json={"prompt": prompt, "model": model}, timeout=60)
        res.raise_for_status()
        data = res.json()
        if "response" in data:
            return data["response"]
        else:
            return f"⚠️ Backend Error: {data.get('error', 'Unknown error')}"
    except requests.exceptions.RequestException as e:
        return f"❌ Network error: {e}"


# Firebase
import firebase_admin
from firebase_admin import credentials, firestore
import pyrebase4 as pyrebase


# === Load Firebase Config ===
with open("firebase_key.json") as f:
    firebase_config = json.load(f)

# Fix newline issue for private key
firebase_config["private_key"] = firebase_config["private_key"].replace("\\n", "\n")

# === Initialize Firestore (Admin SDK) ===
try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    st.success("✅ Firestore connected successfully!")
except Exception as e:
    st.error(f"❌ Firestore initialization error: {e}")

# === Initialize Pyrebase (Auth + Realtime DB) ===
pyrebase_config = {
    "apiKey": "AIzaSyBxGqGP3pC8sDpZw72pv1xZdPQWwenDFSE",
    "authDomain": "careercraft-ai-223d4.firebaseapp.com",
    "projectId": "careercraft-ai-223d4",
    "storageBucket": "careercraft-ai-223d4.appspot.com",  # ✅ fixed domain
    "messagingSenderId": "21816901435",
    "appId": "1:21816901435:web:fab65a12c37f96b755411e",
    "measurementId": "G-42FH9Z2PH0",
    "databaseURL": "https://careercraft-ai-223d4-default-rtdb.firebaseio.com"
}

try:
    firebase_client = pyrebase.initialize_app(pyrebase_config)
    auth = firebase_client.auth()
    st.success("✅ Pyrebase (Auth + Realtime DB) connected successfully!")
except Exception as e:
    st.error(f"❌ Pyrebase initialization error: {e}")

# ---------------------------------
# 🔐 AUTHENTICATION SECTION
# ---------------------------------
## 🔐 AUTH SECTION (always visible)
st.sidebar.header("🔐 User Authentication")
choice = st.sidebar.radio("Login / Signup", ["Login", "Signup"])

email = st.sidebar.text_input("Email", key="email_input")
password = st.sidebar.text_input("Password", type="password", key="password_input")

if choice == "Signup":
    if st.sidebar.button("Create Account", key="signup_button"):
        try:
            user = auth.create_user_with_email_and_password(email, password)
            st.success("✅ Account created successfully! Please log in.")
        except Exception as e:
            st.error(f"Registration failed: {e}")

elif choice == "Login":
    if st.sidebar.button("Login", key="login_button"):
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            st.session_state["user"] = user
            st.session_state["email"] = email
            st.success(f"Welcome back, {email}!")
        except Exception as e:
            st.error(f"Login failed: {e}")

# Logout logic
if "user" in st.session_state:
    if st.sidebar.button("Logout", key="logout_button"):
        del st.session_state["user"]
        del st.session_state["email"]
        st.success("Logged out successfully.")
        st.stop()

# 🚫 Block entire main app before login
if "user" not in st.session_state:
    st.markdown("### 👆 Please log in or sign up to access CareerCraft AI.")
    st.stop()

# 🚀 MAIN APP (renders only once after login)
st.markdown("### ✅ Firestore connected successfully!")
st.markdown("### ✅ Pyrebase (Auth + Realtime DB) connected successfully!")
st.markdown(f"### 👋 Welcome, **{st.session_state['email']}!**")
st.markdown("---")

# 🧠 Placeholder for next sections (e.g., dashboard, tailoring UI, analytics)
st.info("🚧 Resume tailoring dashboard coming next...")



# ---------------- Load environment ----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GEN_API_KEY")

# ---------------- Streamlit config ----------------
st.set_page_config(page_title="CareerCraft AI — Resume Copilot & Chat", layout="wide")

# ---------------- CSS / Small UI polish ----------------
st.markdown(
    """
<style>
body {background: radial-gradient(circle at 10% 20%, #001219 0%, #005f73 90%); color: #fff;}
h1,h2,h3,h4,h5 {color: #94d2bd;}
.stButton>button {background: linear-gradient(90deg, #0a9396, #94d2bd); color: white; border: none; border-radius: 10px; font-weight: bold; padding: 10px 20px; transition: 0.3s;}
.stButton>button:hover {transform: scale(1.05); background: linear-gradient(90deg, #94d2bd, #0a9396);}
.card {background: rgba(255, 255, 255, 0.1); border: 1px solid rgba(255,255,255,0.15); backdrop-filter: blur(12px); border-radius: 12px; padding: 18px; margin: 10px 0;}
.badge {display:inline-block; padding:6px 10px; border-radius:5px; margin:3px; color:white; font-size:13px; font-weight:500;}
.badge-green {background-color:#2a9d8f;}
.badge-red {background-color:#e63946;}
.diff-added {background-color:#2a9d8f33; padding:2px 6px; margin:2px 0;}
.diff-removed {background-color:#e6394633; padding:2px 6px; margin:2px 0;}
.metric-card {background: rgba(255,255,255,0.07); padding: 15px; border-radius: 10px; text-align: center; font-size: 15px; border: 1px solid rgba(255,255,255,0.2);}
.metric-card h3 {margin:0;font-size:16px;color:#94d2bd;}
.metric-card p {margin:0;font-size:22px;color:#fff;font-weight:bold;}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- Constants ----------------
CACHE_FILE = "cache.json"
FIREBASE_KEY_FILE = "firebase_key.json"

LEARNING_MAP = {
    "Python": "Kaggle: Python for Data Science / YouTube Intro to Python",
    "React": "FreeCodeCamp React Course / YouTube React Tutorial",
    "AWS": "AWS Free Tier Training / A Cloud Guru intro videos",
    "Data Analysis": "Kaggle Learn: Data Analysis / Pandas tutorials",
    "Machine Learning": "Kaggle: Intro to ML / Coursera ML (audit)",
    "SQL": "Mode Analytics SQL Tutorial / Khan Academy",
    "Docker": "Docker docs + YouTube crash course",
    "Communication": "Coursera: Effective Communication / YouTube",
}

CAREER_MAP = {
    "Python": ["Data Analyst", "ML Engineer"],
    "React": ["Frontend Developer", "Fullstack Developer"],
    "AWS": ["DevOps Engineer", "Cloud Engineer"],
    "Data Analysis": ["Data Analyst", "Business Analyst"],
    "Machine Learning": ["ML Engineer", "Data Scientist"],
    "SQL": ["Data Analyst", "BI Developer"],
    "Docker": ["DevOps Engineer"],
}

VALID_SKILLS = list(CAREER_MAP.keys())

CURATED_PLATFORMS = {
    "Kaggle": "https://www.kaggle.com/learn/",
    "Coursera": "https://www.coursera.org/courses?query=",
    "FreeCodeCamp": "https://www.freecodecamp.org/learn/",
    "YouTubePreset": [
        ("Python Full Course for Beginners", "https://www.youtube.com/watch?v=_uQrJ0TkZlc"),
        ("React Tutorial for Beginners", "https://www.youtube.com/watch?v=SqcY0GlETPk"),
        ("Machine Learning Crash Course", "https://www.youtube.com/watch?v=7eh4d6sabA0"),
    ],
}

# Register nicer font for PDFs if available
try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", "DejaVuSans.ttf"))
    DEFAULT_FONT = "DejaVuSans"
except Exception:
    DEFAULT_FONT = "Helvetica"

# ---------------- Helper utilities ----------------
def save_cache(data):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Could not write cache: {e}")


def load_cache():
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def make_key(resume_text, job_desc, extra=""):
    token = (resume_text or "") + "---" + (job_desc or "") + "---" + extra
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def read_pdf(file):
    try:
        reader = PdfReader(file)
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    except Exception:
        return ""


def read_docx(file):
    try:
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""


def extract_keywords_tfidf(*texts, top_n=10):
    corpus = [t if t and t.strip() else " " for t in texts]
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=500)
        X = vec.fit_transform(corpus)
        feature_names = vec.get_feature_names_out()
        scores = X.toarray()
        keywords = []
        for row in scores:
            items = sorted(zip(feature_names, row), key=lambda x: x[1], reverse=True)
            keywords.append([w for w, s in items[:top_n] if s > 0])
        return keywords
    except Exception:
        return [[] for _ in texts]


def split_job_descriptions(jd_text):
    if not jd_text:
        return [""]
    roles = [r.strip() for r in jd_text.split("Our Offering") if r.strip()]
    return roles if roles else [jd_text]


def extract_skills(text):
    keywords = [
        "Python", "Java", "React", "Spring Boot", "AWS", "Azure", "GCP",
        "SQL", "MongoDB", "Machine Learning", "DevOps", "CI/CD",
        "JavaScript", "HTML", "CSS", "ServiceNow", "Data Science", "Docker",
        "Communication", "Leadership", "Data Analysis", "Cloud"
    ]
    found = [kw for kw in keywords if kw.lower() in (text or "").lower()]
    return list(dict.fromkeys(found))


def compute_fit_score(selected_skills, jd_skills):
    selected = set([s.lower() for s in (selected_skills or [])])
    jd = set([s.lower() for s in (jd_skills or [])])
    if jd:
        matched = len(selected.intersection(jd))
        score = int(round(100 * (matched / len(jd)))) if len(jd) > 0 else 0
        return max(0, min(100, score))
    elif selected:
        return min(95, 50 + 10 * len(selected))
    else:
        return 0


# ---------------- Resume tailoring core ----------------
def generate_tailored_resume(resume_text, job_desc, extracted_skills=None, style="Action-Oriented", mock_mode=False):
    cache = load_cache()
    key = make_key(resume_text or "", job_desc or "", ",".join(extracted_skills or []) + style)
    if key in cache:
        return cache[key]

    jd_skills = extract_skills(job_desc or "")

    gpt_text = None
    if not (mock_mode or not OPENAI_API_KEY):
        prompt = f"Tailor the following resume text to the job description in a {style} style. Emphasize skills and style appropriately.\n\nResume:\n{resume_text}\n\nJob Description:\n{job_desc}"
        try:
            gpt_text = get_ai_response(prompt)
        except Exception as e:
            gpt_text = f"[AI generation failed: {e}]\n\nPlease try again or enable mock mode."

    if gpt_text is None:
        skills_pool = extracted_skills or ["Python", "React", "AWS", "SQL", "Data Analysis"]
        selected_skills = random.sample(skills_pool, min(3, len(skills_pool)))
        missing_skills = [s for s in (jd_skills or []) if s not in selected_skills][:2]
        fit_score = compute_fit_score(selected_skills, jd_skills)
        learning_paths = [LEARNING_MAP.get(s, f"Learn {s} online") for s in missing_skills]
        feedback = []
        if style == "Action-Oriented":
            feedback = [f"Show measurable results for {s}." for s in selected_skills]
        elif style == "Data-Driven":
            feedback = [f"Include metrics and analytics experience for {s}." for s in selected_skills]
        elif style == "Leadership-Focused":
            feedback = [f"Highlight leadership or team contributions in {s}." for s in selected_skills]
        version_text = (
            f"[{style} Tailored Resume]\n\nTop Skills: {', '.join(selected_skills)}\n\nFit Score: {fit_score}/100\n\n"
            f"Skill Gaps: {', '.join(missing_skills)}\n\n" + "\n".join(feedback)
        )
        result = {
            "version_text": version_text,
            "selected_skills": selected_skills,
            "fit_score": fit_score,
            "missing_skills": missing_skills,
            "learning_paths": learning_paths,
            "style": style,
            "feedback": feedback,
        }
        cache[key] = result
        save_cache(cache)
        return result

    candidate_skills = extracted_skills or extract_skills(resume_text or "")
    if jd_skills:
        selected_skills = [s for s in candidate_skills if s in jd_skills]
    else:
        selected_skills = list(candidate_skills)[:3]
    if not selected_skills:
        selected_skills = list(candidate_skills)[:3] or (extracted_skills or [])[:3]

    missing_skills = [s for s in (jd_skills or []) if s not in selected_skills]
    fit_score = compute_fit_score(selected_skills, jd_skills)

    result = {
        "version_text": gpt_text,
        "selected_skills": selected_skills,
        "fit_score": fit_score,
        "missing_skills": missing_skills,
        "learning_paths": [LEARNING_MAP.get(s, f"Learn {s} online") for s in missing_skills],
        "style": style,
        "feedback": [],
    }
    cache[key] = result
    save_cache(cache)
    return result


def predict_career_paths(skills):
    roles = []
    for s in skills:
        for r in CAREER_MAP.get(s, []):
            if r not in roles:
                roles.append(r)
    return roles[:5]


# ---------------- Learning resources assembly ----------------
def assemble_learning_resources(skills):
    skills = list(dict.fromkeys(skills))
    final = {}
    try:
        fetched = get_learning_resources(skills) or {}
        for s in skills:
            links = []
            if s in fetched and fetched[s]:
                links.extend(fetched[s])
            links.append(f"[Kaggle: {s}]({CURATED_PLATFORMS['Kaggle']}{s.replace(' ', '-')})")
            links.append(f"[Coursera: {s}]({CURATED_PLATFORMS['Coursera']}{s})")
            links.append(f"[FreeCodeCamp: {s}]({CURATED_PLATFORMS['FreeCodeCamp']})")
            for title, url in CURATED_PLATFORMS["YouTubePreset"]:
                if s.lower() in title.lower() or ("python" in title.lower() and s.lower() == "python"):
                    links.append(f"[🎥 {title}]({url})")
            seen = set()
            deduped = []
            for l in links:
                if l not in seen:
                    deduped.append(l)
                    seen.add(l)
            final[s] = deduped
        if not final or all(not v for v in final.values()):
            raise Exception("No resources returned from get_learning_resources()")
        return final
    except Exception:
        fallback = {}
        for s in skills:
            fallback[s] = [
                f"[Kaggle: {s}]({CURATED_PLATFORMS['Kaggle']}{s.replace(' ', '-')})",
                f"[Coursera: {s}]({CURATED_PLATFORMS['Coursera']}{s})",
                f"[FreeCodeCamp: {s}]({CURATED_PLATFORMS['FreeCodeCamp']})",
            ]
            for title, url in CURATED_PLATFORMS["YouTubePreset"]:
                if s.lower() in title.lower() or ("python" in title.lower() and s.lower() == "python"):
                    fallback[s].insert(0, f"[🎥 {title}]({url})")
        return fallback


# ---------------- Modern PDF creator ----------------
def _draw_wrapped_text(pdf, text, x, y, max_width, leading=14, fontname=DEFAULT_FONT, fontsize=11):
    words = text.split()
    line = ""
    for w in words:
        test = f"{line} {w}".strip()
        width = pdf.stringWidth(test, fontname, fontsize)
        if width <= max_width:
            line = test
        else:
            pdf.drawString(x, y, line)
            y -= leading
            line = w
    if line:
        pdf.drawString(x, y, line)
        y -= leading
    return y


def create_pdf_modern(v_text, candidate_name, version_label, missing_skills, selected_skills=None):
    buf = BytesIO()
    pdf = canvas.Canvas(buf, pagesize=letter)
    w, h = letter
    margin = 50
    usable_width = w - 2 * margin
    x = margin
    y = h - 60

    pdf.setFont(DEFAULT_FONT, 18)
    pdf.setFillColor(colors.HexColor("#0a9396"))
    pdf.drawString(x, y, f"{candidate_name} — {version_label}")
    y -= 28

    pdf.setStrokeColor(colors.HexColor("#94d2bd"))
    pdf.setFillColor(colors.black)
    pdf.setFont(DEFAULT_FONT, 10)
    pdf.setFillColor(colors.HexColor("#94d2bd"))
    pdf.drawString(x, y, "Generated by CareerCraft AI — Resume Copilot")
    y -= 20
    pdf.setFillColor(colors.black)

    fit_score = None
    try:
        for line in (v_text or "").splitlines():
            if "Fit Score" in line:
                m = re.search(r"(\d{1,3})", line)
                if m:
                    fit_score = int(m.group(1))
                    break
    except Exception:
        fit_score = None

    pdf.setFont(DEFAULT_FONT, 12)
    fs = fit_score if fit_score is not None else ""
    pdf.setFillColor(colors.HexColor("#f4f4f4"))
    pdf.rect(x, y - 10, 120, 36, fill=True, stroke=False)
    pdf.setFillColor(colors.HexColor("#001219"))
    pdf.setFont(DEFAULT_FONT, 12)
    pdf.drawString(x + 8, y + 8, "Fit Score")
    pdf.setFont(DEFAULT_FONT, 18)
    pdf.drawString(x + 8, y - 6, f"{fs}%")

    top_skills = ", ".join(selected_skills or []) if selected_skills else ""
    pdf.setFillColor(colors.HexColor("#f4f4f4"))
    pdf.rect(x + 140, y - 10, usable_width - 140, 36, fill=True, stroke=False)
    pdf.setFillColor(colors.HexColor("#001219"))
    pdf.setFont(DEFAULT_FONT, 12)
    pdf.drawString(x + 148, y + 8, "Top Skills")
    pdf.setFont(DEFAULT_FONT, 11)
    pdf.drawString(x + 148, y - 6, top_skills)
    y -= 50

    pdf.setFont(DEFAULT_FONT, 12)
    pdf.setFillColor(colors.HexColor("#94d2bd"))
    pdf.drawString(x, y, "Skill Gaps:")
    pdf.setFillColor(colors.black)
    y -= 18
    if missing_skills:
        for s in missing_skills:
            y = _draw_wrapped_text(pdf, f"• {s}", x + 8, y, usable_width - 16, leading=14)
    else:
        y = _draw_wrapped_text(pdf, "• None identified.", x + 8, y, usable_width - 16, leading=14)
    y -= 6

    pdf.setFont(DEFAULT_FONT, 12)
    pdf.setFillColor(colors.HexColor("#94d2bd"))
    pdf.drawString(x, y, "Learning Resources:")
    pdf.setFillColor(colors.black)
    y -= 18
    skills_to_use = missing_skills or (selected_skills or [])
    if skills_to_use:
        try:
            resources = assemble_learning_resources(skills_to_use)
        except Exception:
            resources = {}
        pdf.setFont(DEFAULT_FONT, 10)
        for skill, links in resources.items():
            pdf.setFillColor(colors.HexColor("#0a9396"))
            pdf.drawString(x + 6, y, f"🔹 {skill}")
            y -= 14
            pdf.setFillColor(colors.black)
            for link in links:
                display = link
                url = None
                if link.startswith("[") and "](" in link and link.endswith(")"):
                    try:
                        display = link.split("](")[0].lstrip("[")
                        url = link.split("](")[1][:-1]
                    except:
                        url = None
                text_to_draw = display
                text_width = pdf.stringWidth(text_to_draw, DEFAULT_FONT, 10)
                if text_width <= usable_width - 40:
                    pdf.drawString(x + 20, y, text_to_draw)
                    if url:
                        pdf.linkURL(url, (x + 20, y - 2, x + 20 + text_width, y + 10), relative=0)
                    y -= 12
                else:
                    y = _draw_wrapped_text(pdf, f"- {text_to_draw}", x + 20, y, usable_width - 40, leading=12)
                    if url:
                        pdf.linkURL(url, (x + 20, y + 2, x + usable_width - 20, y + 18), relative=0)
                    y -= 4
                if y < 100:
                    pdf.showPage()
                    y = h - 60
                    pdf.setFont(DEFAULT_FONT, 10)
    else:
        pdf.drawString(x + 8, y, "No learning resources available.")
        y -= 14

    y -= 8
    pdf.setStrokeColor(colors.HexColor("#e6e6e6"))
    pdf.line(x, y, w - margin, y)
    y -= 18
    pdf.setFont(DEFAULT_FONT, 12)
    pdf.setFillColor(colors.HexColor("#94d2bd"))
    pdf.drawString(x, y, "Full Resume Content:")
    pdf.setFillColor(colors.black)
    y -= 16
    pdf.setFont(DEFAULT_FONT, 10)

    for paragraph in (v_text or "").split("\n\n"):
        lines = paragraph.splitlines()
        for line in lines:
            if not line.strip():
                y -= 10
                continue
            y = _draw_wrapped_text(pdf, line.strip(), x, y, usable_width, leading=12)
            if y < 100:
                pdf.showPage()
                y = h - 60
                pdf.setFont(DEFAULT_FONT, 10)
        y -= 8
        if y < 120:
            pdf.showPage()
            y = h - 60
            pdf.setFont(DEFAULT_FONT, 10)

    pdf.save()
    buf.seek(0)
    return buf


# ---------------- Firestore init & helpers ----------------
def init_firestore(show_success=True):
    try:
        if os.path.exists(FIREBASE_KEY_FILE):
            if not firebase_admin._apps:
                cred = credentials.Certificate(FIREBASE_KEY_FILE)
                firebase_admin.initialize_app(cred)
        else:
            if not firebase_admin._apps:
                try:
                    firebase_admin.initialize_app()
                except Exception:
                    pass

        db_client = None
        try:
            db_client = firestore.client()
            # lightweight connectivity test
            db_client.collection("test_connection").add({"status": "connected", "ts": firestore.SERVER_TIMESTAMP})
            if show_success:
                st.success("✅ Firestore connected successfully!")
        except Exception as e:
            if show_success:
                st.error(f"❌ Firestore connection failed: {e}")
        return db_client
    except Exception as outer:
        if show_success:
            st.error(f"❌ Firestore initialization error: {outer}")
        return None


db = init_firestore(show_success=True)

# read apiKey from firebase_key.json for REST auth if possible
FIREBASE_API_KEY = None
if os.path.exists(FIREBASE_KEY_FILE):
    try:
        with open(FIREBASE_KEY_FILE, "r", encoding="utf-8") as f:
            fb_conf = json.load(f)
            # firebase_config sometimes nested; try common places
            FIREBASE_API_KEY = fb_conf.get("apiKey") or fb_conf.get("web", {}).get("apiKey") or fb_conf.get("project_info", {}).get("api_key", [None])[0]
    except Exception:
        FIREBASE_API_KEY = None

# ---------------- Firestore save helper (user-aware) ----------------
def save_to_firestore(resume_name, jd_name, extracted_skills, fit_score, suggestions):
    try:
        if db:
            payload = {
                "timestamp": datetime.now().isoformat(),
                "resume_name": resume_name,
                "job_description_name": jd_name,
                "extracted_skills": extracted_skills,
                "fit_score": fit_score,
                "suggestions": suggestions
            }
            # if user logged in, add user email
            user = st.session_state.get("user")
            if user:
                payload["user_email"] = user.get("email")
                payload["user_uid"] = user.get("uid")
            db.collection("results").add(payload)
            st.success("📊 Results saved to Firestore successfully!")
        else:
            st.warning("⚠️ Firestore not initialized.")
    except Exception as e:
        st.warning(f"⚠️ Could not save to Firestore: {e}")


# ---------------- Firebase Auth: register/login/logout ----------------
def firebase_register(email: str, password: str):
    """
    Create user using admin SDK (server side). Returns dict or raises.
    """
    try:
        user = fb_auth.create_user(email=email, password=password)
        return {"uid": user.uid, "email": user.email}
    except Exception as e:
        raise e


def firebase_login(email: str, password: str):
    """
    Sign in with password using Firebase REST API -> returns idToken, localId (uid)
    Requires FIREBASE_API_KEY to be set (from firebase_key.json).
    """
    if not FIREBASE_API_KEY:
        raise Exception("Firebase API key not found in firebase_key.json")
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    resp = requests.post(url, json=payload, timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        # data contains idToken, refreshToken, expiresIn, localId, email
        return {"idToken": data.get("idToken"), "refreshToken": data.get("refreshToken"), "uid": data.get("localId"), "email": data.get("email")}
    else:
        raise Exception(resp.json().get("error", {}).get("message", resp.text))


def firebase_logout():
    # client-side simple: remove session state
    if "user" in st.session_state:
        del st.session_state["user"]


# ---------------- Sidebar Auth UI and recent history ----------------
def auth_ui_sidebar():
    st.markdown("## 🔐 Account")
    if st.session_state.get("user"):
        user = st.session_state["user"]
        st.markdown(f"**Signed in as:** {user.get('email')}")
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🔓 Logout"):
                firebase_logout()
                st.success("Logged out.")
                st.experimental_rerun()
        with col2:
            st.write("")  # space
        # Show recent history for this user
        if db:
            try:
                q = db.collection("results").where("user_email", "==", user.get("email")).order_by("timestamp", direction=firestore.Query.DESCENDING).limit(5)
                docs = q.stream()
                st.markdown("### 🕒 Your recent tailoring")
                for d in docs:
                    data = d.to_dict()
                    ts = data.get("timestamp", "")
                    st.markdown(f"- **{data.get('resume_name','-')}** | Fit: {data.get('fit_score','-')}% | {ts[:19]}")
            except Exception as e:
                st.warning(f"⚠️ Unable to load your history: {e}")
    else:
        st.markdown("### 🔐 Login or Register")
        email = st.text_input("Email", key="auth_email")
        password = st.text_input("Password", type="password", key="auth_pass")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login"):
                if not email or not password:
                    st.warning("Enter email & password.")
                else:
                    try:
                        res = firebase_login(email, password)
                        st.session_state["user"] = {"email": res["email"], "uid": res["uid"], "idToken": res["idToken"]}
                        st.success(f"Welcome back, {email}!")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Login failed: {e}")
        with col2:
            if st.button("Register"):
                if not email or not password:
                    st.warning("Enter email & password to register.")
                else:
                    try:
                        created = firebase_register(email, password)
                        # after create, sign-in to get token
                        res = firebase_login(email, password)
                        st.session_state["user"] = {"email": res["email"], "uid": res["uid"], "idToken": res["idToken"]}
                        st.success("Account created and signed in.")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Registration failed: {e}")


# ---------------- Sidebar (controls + chat + auth) ----------------
with st.sidebar:
    st.markdown("## ⚡ CareerCraft AI")
    st.markdown("**Your AI-powered resume tailoring copilot.**")
    # AUTH UI
    auth_ui_sidebar()
    st.markdown("---")

    # Controls
    candidate_name = st.text_input("👤 Candidate Name", "Candidate")
    num_versions = st.slider("🧬 Versions", 1, 3, 2)
    resume_style = st.selectbox("🧠 Resume Style", ["Action-Oriented", "Data-Driven", "Leadership-Focused"])
    mock_mode_toggle = st.checkbox("🧩 Force Mock Mode", value=False)
    st.markdown("---")
    st.markdown("## 💬 AI Chat Bot")
    user_prompt = st.text_area("Ask CareerCraft AI:", height=120)
    if st.button("Send Prompt"):
        if user_prompt.strip():
            with st.spinner("Generating AI response..."):
                try:
                    bot_reply = get_ai_response(user_prompt)
                except Exception as e:
                    bot_reply = f"[AI error: {e}]"
                st.markdown(f"**AI Response:** {bot_reply}")


# ---------------- Main UI: Resume Tailoring ----------------
st.markdown("## 🚀 Resume Tailoring")

col1, col2 = st.columns(2)

with col1:
    upload_resume = st.file_uploader("📄 Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
    if upload_resume:
        resume_text = read_pdf(upload_resume) if upload_resume.name.endswith(".pdf") else read_docx(upload_resume)
        st.success(f"✅ Loaded {upload_resume.name}")
    else:
        resume_text = st.text_area("Or Paste Resume Text", height=280)

with col2:
    upload_jd = st.file_uploader("💼 Upload Job Description (PDF/DOCX)", type=["pdf", "docx"])
    if upload_jd:
        job_desc = read_pdf(upload_jd) if upload_jd.name.endswith(".pdf") else read_docx(upload_jd)
        st.success(f"✅ Loaded {upload_jd.name}")
    else:
        job_desc = st.text_area("Or Paste Job Description", height=280)

# ✅ Proceed only when both inputs exist
if resume_text and job_desc:
    st.success("✅ Both resume and job description loaded successfully!")
    # 🔥 Continue tailoring logic here (AI analysis, matching, etc.)
else:
    st.info("👆 Please upload both files or paste both texts to proceed.")

# ---------------- Extract skills ----------------
resume_kw, jd_kw = extract_keywords_tfidf(resume_text or "", job_desc or "", top_n=15)
all_extracted_skills = [s for s in list(dict.fromkeys([*(resume_kw or []), *(jd_kw or [])])) if s in VALID_SKILLS]
if not all_extracted_skills:
    fallback_skills = extract_skills((resume_text or "") + " " + (job_desc or ""))
    all_extracted_skills = [s for s in fallback_skills if s in VALID_SKILLS]

st.markdown("### 🧩 Extracted Skills")
st.markdown(f"<div class='card'>{', '.join(all_extracted_skills) if all_extracted_skills else 'No skills detected.'}</div>", unsafe_allow_html=True)

# ---------------- Tailor resume action ----------------
tailored_versions = []
mock_mode = mock_mode_toggle or (not OPENAI_API_KEY)

if st.button("🎯 Tailor Resume Now"):
    if not (resume_text and resume_text.strip()) or not (job_desc and job_desc.strip()):
        st.warning("Please enter both resume and job description.")
    else:
        job_roles = split_job_descriptions(job_desc)
        for role_text in job_roles:
            for i in range(num_versions):
                with st.spinner(f"Generating version {i+1}..."):
                    v = generate_tailored_resume(resume_text, role_text, all_extracted_skills, resume_style, mock_mode)
                    v.setdefault("selected_skills", all_extracted_skills[:3])
                    v.setdefault("missing_skills", [])
                    v.setdefault("feedback", [])
                    v.setdefault("fit_score", v.get("fit_score", 0))
                    tailored_versions.append(v)

# ---------------- Display results ----------------
if tailored_versions:
    latest = tailored_versions[-1]

    # Save latest tailoring summary to Firestore (user-aware)
    save_to_firestore(
        resume_name=upload_resume.name if upload_resume else "manual_resume",
        jd_name=upload_jd.name if upload_jd else "manual_jd",
        extracted_skills=latest.get("selected_skills", []),
        fit_score=latest.get("fit_score", 0),
        suggestions=latest.get("feedback", [])
    )

    st.markdown("## 📊 Resume Insights Dashboard")

    cols = st.columns(4)
    cols[0].markdown(f"<div class='metric-card'><h3>Fit Score</h3><p>{latest.get('fit_score', 0)}%</p></div>", unsafe_allow_html=True)
    cols[1].markdown(f"<div class='metric-card'><h3>Top Skills</h3><p>{', '.join(latest.get('selected_skills', []))}</p></div>", unsafe_allow_html=True)
    cols[2].markdown(f"<div class='metric-card'><h3>Skill Gaps</h3><p>{', '.join(latest.get('missing_skills', []))}</p></div>", unsafe_allow_html=True)
    cols[3].markdown(f"<div class='metric-card'><h3>Style</h3><p>{latest.get('style', '')}</p></div>", unsafe_allow_html=True)

    tabs = st.tabs([f"{v.get('style','Style')} v{i+1}" for i, v in enumerate(tailored_versions)])
    for i, (tab, v) in enumerate(zip(tabs, tailored_versions)):
        with tab:
            st.markdown(f"### 🧠 {v.get('style','Style')} Version {i+1}")
            st.markdown(f"**Fit Score:** {v.get('fit_score', 0)}%")

            for s in v.get('selected_skills', []):
                st.markdown(f"<span class='badge badge-green'>{s}</span>", unsafe_allow_html=True)
            for s in v.get('missing_skills', []):
                st.markdown(f"<span class='badge badge-red'>{s}</span>", unsafe_allow_html=True)

            st.markdown("#### Feedback")
            for f in v.get('feedback', []):
                st.markdown(f"- {f}")

            st.markdown("#### 📚 Learning Resources")
            skills_for_resources = v.get('missing_skills') or v.get('selected_skills') or all_extracted_skills
            if skills_for_resources:
                with st.spinner("Fetching learning resources..."):
                    resources = assemble_learning_resources(skills_for_resources)
                    for skill, links in resources.items():
                        st.markdown(f"**{skill}**")
                        for link in links:
                            st.markdown(f"- {link}", unsafe_allow_html=True)
            else:
                st.info("No missing skills and no selected skills detected.")

            updated_text = st.text_area(f"📝 Resume Content v{i+1}", v.get("version_text", ""), height=250)
            diff_html = ""
            for line in difflib.ndiff(v.get("version_text", "").splitlines(), updated_text.splitlines()):
                if line.startswith("+"):
                    diff_html += f"<div class='diff-added'>{line}</div>"
                elif line.startswith("-"):
                    diff_html += f"<div class='diff-removed'>{line}</div>"
            st.markdown("#### 🔍 Change Diff")
            st.markdown(diff_html, unsafe_allow_html=True)

            paths = predict_career_paths(v.get('selected_skills', []))
            st.markdown("#### 🚀 Suggested Career Paths")
            st.markdown(", ".join(paths) if paths else "No paths detected.")

            # PDF export (modern)
            pdf_buf = create_pdf_modern(updated_text, candidate_name, f"{v.get('style','Style')} v{i+1}", v.get('missing_skills', []), v.get('selected_skills', []))
            st.download_button(f"📄 Download PDF v{i+1}", pdf_buf, file_name=f"{candidate_name}_{v.get('style','style')}_v{i+1}.pdf")
