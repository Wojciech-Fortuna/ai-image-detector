import os
import requests
from tqdm import tqdm

OUTPUT_DIR = "met_art"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEARCH_TERMS = [
    "drawing",
    "painting",
    "watercolor",
    "sketch",
    "ink",
    "illustration"
]

limit = 300


def search_objects(query):
    url = (
        "https://collectionapi.metmuseum.org/public/collection/v1/search"
        f"?q={query}&hasImages=true"
    )
    try:
        data = requests.get(url).json()
        return data.get("objectIDs", [])
    except Exception:
        return []


object_ids = set()

print("[INFO] Fetching object lists by keywords...")
for term in SEARCH_TERMS:
    print(f"[INFO] Searching for: {term}")
    ids = search_objects(term)
    if ids:
        object_ids.update(ids)

object_ids = list(object_ids)
print(
    f"[INFO] After filtering, found {len(object_ids)} potential objects."
)

downloaded = 0

for object_id in tqdm(object_ids):
    if downloaded >= limit:
        break

    url = (
        "https://collectionapi.metmuseum.org/public/collection/v1/objects/"
        f"{object_id}"
    )

    try:
        data = requests.get(url).json()
    except Exception:
        continue

    img_url = data.get("primaryImage")
    if not img_url:
        continue

    classification = data.get("classification", "")
    if classification not in [
        "Paintings",
        "Drawings",
        "Prints",
        "Watercolors",
    ]:
        continue

    filename = f"{object_id}.jpg"
    file_path = os.path.join(OUTPUT_DIR, filename)

    try:
        img_data = requests.get(img_url).content
        with open(file_path, "wb") as f:
            f.write(img_data)
        downloaded += 1
    except Exception:
        continue

print(
    f"[DONE] Downloaded {downloaded} images into folder '{OUTPUT_DIR}'."
)
