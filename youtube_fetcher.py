# youtube_fetcher.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

def get_learning_resources(skills):
    """
    Fetch top 3 YouTube tutorials for each skill using YouTube Data API.
    If API fails or key is missing, return {} and let fallback handle it in app.py.
    """
    results = {}
    if not YOUTUBE_API_KEY:
        print("⚠️ No YOUTUBE_API_KEY found. Returning empty results (fallback will handle).")
        return {}

    for skill in skills:
        try:
            query = f"{skill} tutorial for beginners"
            url = (
                f"https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=3"
                f"&q={query}&key={YOUTUBE_API_KEY}&type=video"
            )
            r = requests.get(url)
            if r.status_code == 200:
                data = r.json()
                videos = [
                    f"[🎥 {item['snippet']['title']}]"
                    f"(https://www.youtube.com/watch?v={item['id']['videoId']})"
                    for item in data.get("items", [])
                ]
                results[skill] = videos
            else:
                results[skill] = []
        except Exception as e:
            print(f"Error fetching YouTube for {skill}: {e}")
            results[skill] = []
    return results
