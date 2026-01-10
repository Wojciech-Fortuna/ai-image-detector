import os
import time
import requests
from typing import Dict, Any, Optional

API_KEY = "PUT_YOUR_KEY_HERE"

queries = ["people", "city", "nature", "animals", "food", "architecture"]

NUM_PER_CATEGORY = 30

IMAGE_SIZE_KEY = "large"

OUT_DIR = "pexels"

PER_PAGE = 80

REQUEST_DELAY = 0.1
MAX_RETRIES = 3
BACKOFF_SECONDS = 2.0


def pexels_get(url: str, headers: Dict[str, str], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(BACKOFF_SECONDS * attempt)
            else:
                resp.raise_for_status()
        except Exception as e:
            last_exc = e
            time.sleep(BACKOFF_SECONDS * attempt)
    if last_exc:
        raise last_exc
    return {}


def download_file(url: str, dest_path: str) -> None:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return
        except Exception as e:
            last_exc = e
            time.sleep(BACKOFF_SECONDS * attempt)
    if last_exc:
        raise last_exc


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    if API_KEY.strip() == "PUT_YOUR_KEY_HERE":
        raise SystemExit("Missing API key. Paste it into the API_KEY variable at the top of the file.")

    headers = {"Authorization": API_KEY}
    ensure_dir(OUT_DIR)

    seen_ids = set()

    for q in queries:
        print(f"\n=== Category: {q} (target: {NUM_PER_CATEGORY}) ===")
        q_dir = os.path.join(OUT_DIR, q.replace(" ", "_"))
        ensure_dir(q_dir)

        downloaded_for_q = 0
        page = 1

        while downloaded_for_q < NUM_PER_CATEGORY:
            remaining = NUM_PER_CATEGORY - downloaded_for_q
            params = {
                "query": q,
                "per_page": min(PER_PAGE, remaining),
                "page": page
            }
            data = pexels_get("https://api.pexels.com/v1/search", headers, params)
            photos = data.get("photos", [])

            if not photos:
                print("No more results for this category (or API limit reached).")
                break

            for photo in photos:
                pid = photo.get("id")
                if pid in seen_ids:
                    print(f"[SKIP DUPLICATE] pid={pid} already seen")
                    continue
                seen_ids.add(pid)

                src = photo.get("src", {})
                img_url = src.get(IMAGE_SIZE_KEY) or src.get("original")
                if not img_url:
                    print(f"[SKIP] Photo {pid} has no usable URL (missing '{IMAGE_SIZE_KEY}' and 'original').")
                    continue

                filename = f"{q.replace(' ', '_')}_{pid}.jpg"
                dest_path = os.path.join(q_dir, filename)

                try:
                    download_file(img_url, dest_path)
                except Exception as e:
                    print(f"Error downloading {img_url}: {e}")
                    continue

                downloaded_for_q += 1
                print(f"Downloaded {downloaded_for_q}/{NUM_PER_CATEGORY} for \"{q}\" -> {filename}")

                if downloaded_for_q >= NUM_PER_CATEGORY:
                    break

            page += 1
            time.sleep(REQUEST_DELAY)

        print(f"Finished \"{q}\": {downloaded_for_q} files (target: {NUM_PER_CATEGORY})")

    print(f"\nDone! All photos saved in folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
