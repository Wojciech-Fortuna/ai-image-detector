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

def preload_openclip():
    import open_clip
    model_name = "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
    _model, _preprocess_train, _preprocess_val = open_clip.create_model_and_transforms(model_name)
    _model.eval()
    _model.to("cpu")

def main():
    print("Preload: importing all methods...")
    import_all_methods()

    print("Preload: torchvision/timm...")
    preload_torchvision_and_timm()

    print("Preload: face_recognition_models...")
    preload_face_recognition_models()

    print("Preload: CLIP...")
    preload_clip()

    print("Preload: OpenCLIP (hf-hub)...")
    preload_openclip()

    print("Preload done.")

if __name__ == "__main__":
    main()
