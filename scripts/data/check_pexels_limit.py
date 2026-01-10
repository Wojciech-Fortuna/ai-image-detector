import time
import requests
from typing import Dict, Any, Optional, Set

API_KEY = "PUT_YOUR_KEY_HERE"
QUERY = "landscape"

PER_PAGE = 80
REQUEST_DELAY = 0.1
MAX_RETRIES = 3
BACKOFF_SECONDS = 2.0


def pexels_get(
    url: str,
    headers: Dict[str, str],
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    last_exception = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=30
            )

            if response.status_code == 200:
                return response.json()

            if response.status_code in (429, 500, 502, 503, 504):
                print(
                    f"Received status {response.status_code}, "
                    f"retrying (attempt {attempt})..."
                )
                time.sleep(BACKOFF_SECONDS * attempt)
            else:
                response.raise_for_status()

        except Exception as e:
            last_exception = e
            print(f"Request error (attempt {attempt}): {e}")
            time.sleep(BACKOFF_SECONDS * attempt)

    if last_exception:
        raise last_exception

    return {}


def main():
    if API_KEY.strip() == "PUT_YOUR_KEY_HERE":
        raise SystemExit(
            "Missing API key. Paste it into the API_KEY variable at the top of the file."
        )

    headers = {"Authorization": API_KEY}

    page = 1
    total_results = None
    total_received = 0
    unique_ids: Set[int] = set()

    print(f'Checking results for query: "{QUERY}"')

    while True:
        params = {
            "query": QUERY,
            "per_page": PER_PAGE,
            "page": page
        }

        data = pexels_get(
            "https://api.pexels.com/v1/search",
            headers,
            params
        )

        if total_results is None:
            total_results = data.get("total_results")
            print(
                f'total_results (from API) for "{QUERY}": {total_results}'
            )

        photos = data.get("photos", [])
        count_this_page = len(photos)

        if count_this_page == 0:
            print("No more results (empty 'photos' list). Finished.")
            break

        print(f"Page {page}: {count_this_page} photos")

        total_received += count_this_page

        for photo in photos:
            photo_id = photo.get("id")
            if photo_id is not None:
                unique_ids.add(photo_id)

        page += 1
        time.sleep(REQUEST_DELAY)

    print("\n=== Summary ===")
    print(f"total_results from first API response: {total_results}")
    print(f"Total photos received (all pages): {total_received}")
    print(f"Number of unique IDs: {len(unique_ids)}")


if __name__ == "__main__":
    main()
