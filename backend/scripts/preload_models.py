import os
import importlib
import pkgutil

os.environ.setdefault("TORCH_HOME", "/app/.cache/torch")
os.environ.setdefault("HF_HOME", "/app/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/app/.cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/app/.cache/huggingface")

def import_all_methods():
    import methods
    for m in pkgutil.iter_modules(methods.__path__):
        importlib.import_module(f"methods.{m.name}")

def preload_torchvision_and_timm():
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
    _ = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)

    import timm
    _ = timm.create_model("convnext_tiny", pretrained=True)

def preload_face_recognition_models():
    import face_recognition_models  # noqa

def preload_clip():
    import clip
    _model, _preprocess = clip.load("ViT-B/32", device="cpu")

def main():
    print("Preload: importing all methods...")
    import_all_methods()

    print("Preload: torchvision/timm...")
    preload_torchvision_and_timm()

    print("Preload: face_recognition_models...")
    preload_face_recognition_models()

    print("Preload: CLIP...")
    preload_clip()

    print("Preload done.")

if __name__ == "__main__":
    main()
