import os
import requests
from urllib.parse import urljoin, urlparse
from html.parser import HTMLParser

BASE_URL = "https://lesc.dinfo.unifi.it/FloreView/Dataset/"
OUTPUT_ROOT = "FloreView_Dataset"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

VISITED = set()


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


def get_links(url):
    print(f"[LIST] {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    parser = LinkParser()
    parser.feed(resp.text)
    return parser.links


def download_file(file_url, local_path):
    if os.path.exists(local_path):
        print(f"[SKIP] {local_path} (already exists)")
        return

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    print(f"[GET ] {file_url} -> {local_path}")
    try:
        with requests.get(file_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        print(f"[ERR ] Error downloading {file_url}: {e}")


def crawl_directory(url, local_root):
    if url in VISITED:
        print(f"[SEEN] {url} (already visited)")
        return
    VISITED.add(url)

    links = get_links(url)

    for href in links:
        if href.startswith("..") or href.startswith("?") or href.startswith("#"):
            continue

        full_url = urljoin(url, href)

        if not full_url.startswith(BASE_URL):
            print(f"[SKIP] {full_url} (outside BASE_URL)")
            continue

        if href.endswith("/"):
            path = urlparse(full_url).path
            subdir_name = os.path.basename(path.rstrip("/"))
            local_subdir = os.path.join(local_root, subdir_name)
            os.makedirs(local_subdir, exist_ok=True)
            crawl_directory(full_url, local_subdir)
        else:
            if href.lower().endswith(IMAGE_EXTENSIONS):
                path = urlparse(full_url).path
                filename = os.path.basename(path)
                local_file_path = os.path.join(local_root, filename)
                download_file(full_url, local_file_path)
            else:
                print(f"[IGN ] {full_url} (not an image)")


if __name__ == "__main__":
    root_dir = os.path.abspath(OUTPUT_ROOT)
    os.makedirs(root_dir, exist_ok=True)

    print(f"Starting image download from:\n  {BASE_URL}\nto directory:\n  {root_dir}\n")

    crawl_directory(BASE_URL, root_dir)

    print("\nFinished crawling FloreView/Dataset directories (images only).")
