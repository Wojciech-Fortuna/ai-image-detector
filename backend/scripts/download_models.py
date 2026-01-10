import os
from huggingface_hub import snapshot_download

REPO_ID = os.environ.get(
    "HF_MODELS_REPO",
    "Wojciech-Fortuna/ai-image-detector-models",
)
REVISION = os.environ.get("HF_REVISION", "main")

LOCAL_DIR = os.environ.get("LOCAL_DIR", "/app/models")

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)

    if os.listdir(LOCAL_DIR):
        print(f"[models] '{LOCAL_DIR}' already exists and is not empty, skipping download.")
        return

    print(f"[models] Downloading {REPO_ID}@{REVISION} -> '{LOCAL_DIR}' (public repo)")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        revision=REVISION,
        local_dir=LOCAL_DIR,
    )
    print("[models] Download complete.")

if __name__ == "__main__":
    main()
