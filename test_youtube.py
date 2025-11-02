from youtube_fetcher import get_learning_resources

# Example test for missing skills
test_skills = ["Python", "React", "Machine Learning"]

print("🔍 Fetching resources for:", test_skills)
resources = get_learning_resources(test_skills)

for skill, links in resources.items():
    print(f"\n📘 {skill} Resources:")
    for link in links:
        print(" -", link)
