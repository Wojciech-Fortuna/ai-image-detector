import os
import sqlite3
from glob import glob
from typing import List, Tuple, Optional, Set

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from joblib import dump
from PIL import Image
from catboost import CatBoostClassifier
from catboost.utils import get_gpu_device_count

from base_models import BASE_MODELS

ROOT_DIR = "data"
USE_LOGIT = True
OUTPUT_PATH = "meta_model.joblib"

FACES_DB_PATH = "faces_cache.sqlite"

FACE_ONLY_MODEL_NAMES: Set[str] = {
    "cnn_stylegan",
    "convnext_stylegan1",
}

NEUTRAL_P_AI_FOR_SKIPPED = 0.5

ADD_IS_FACE_FEATURE = True

LABEL_MAP = {"real": 0, "ai": 1}


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return float(np.log(p / (1.0 - p)))


def discover_subdatasets(root_dir: str):
    return [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("imagenet_")
    ]


def collect_image_paths(root_dir: str, split: str):
    subdatasets = discover_subdatasets(root_dir)
    all_paths = []
    all_labels = []

    for sub in subdatasets:
        base = os.path.join(root_dir, sub, split)
        ai_dir = os.path.join(base, "ai")
        real_dir = os.path.join(base, "nature")

        if not os.path.isdir(ai_dir) or not os.path.isdir(real_dir):
            print(f"[WARN] Skipping {sub}/{split}, missing ai/ or nature/")
            continue

        ai_files = glob(os.path.join(ai_dir, "*"))
        real_files = glob(os.path.join(real_dir, "*"))

        all_paths.extend(real_files)
        all_labels.extend([LABEL_MAP["real"]] * len(real_files))
        all_paths.extend(ai_files)
        all_labels.extend([LABEL_MAP["ai"]] * len(ai_files))

    print(f"[INFO] Split '{split}': {len(all_paths)} images")
    return all_paths, np.array(all_labels, dtype=np.int32)


def load_face_flags(
    db_path: str,
    root_dir: str,
    img_paths: List[str],
    default_if_missing: int = 0,
) -> np.ndarray:
    root_dir = os.path.abspath(root_dir)
    rel_paths = [os.path.relpath(p, root_dir).replace("\\", "/") for p in img_paths]

    conn = sqlite3.connect(db_path, timeout=30)
    try:
        out = np.full(len(rel_paths), default_if_missing, dtype=np.int32)

        B = 900
        for start in range(0, len(rel_paths), B):
            chunk = rel_paths[start : start + B]
            qmarks = ",".join(["?"] * len(chunk))
            rows = conn.execute(
                f"SELECT path, contains_face FROM faces_cache WHERE path IN ({qmarks});",
                chunk,
            ).fetchall()
            mapping = {p: int(v) for p, v in rows}
            for i, rp in enumerate(chunk, start=start):
                out[i] = mapping.get(rp, default_if_missing)

        return out
    finally:
        conn.close()


def compute_features_for_images(
    img_paths: List[str],
    models,
    face_flags: np.ndarray,
    face_only_mask: np.ndarray,
    use_logit: bool = True,
    neutral_p: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(img_paths)
    m = len(models)
    X = np.zeros((n, m), dtype=np.float32)
    P_ai = np.zeros((n, m), dtype=np.float32)

    neutral_feat = logit(neutral_p) if use_logit else neutral_p

    for i, img_path in enumerate(img_paths):
        if (i + 1) % 100 == 0 or i == n - 1:
            print(f"[INFO] Processing image {i+1}/{n}: {img_path}")

        with Image.open(img_path) as img:
            img = img.copy()

        has_face = int(face_flags[i]) == 1

        for j, method in enumerate(models):
            if (not has_face) and bool(face_only_mask[j]):
                P_ai[i, j] = float(neutral_p)
                X[i, j] = float(neutral_feat)
                continue

            result = method.analyze(img, score_only=True)
            p_ai = float(result.score)

            if not np.isfinite(p_ai):
                print("\n" + "=" * 80)
                print("[ERROR] Non-finite score detected!")
                print(f"Image index:    {i}")
                print(f"Image path:     {img_path}")
                print(f"Method index:   {j}")
                print(f"Method name:    {getattr(method, 'name', 'unknown')}")
                print(f"Returned score: {p_ai}")
                try:
                    print(f"Metrics:        {result.metrics}")
                except Exception:
                    print("Metrics:        <unavailable>")
                print("=" * 80 + "\n")
                p_ai = neutral_p

            if p_ai < 0.0 or p_ai > 1.0:
                print(
                    f"[WARN] Out-of-range score {p_ai} from method "
                    f"'{getattr(method, 'name', 'unknown')}' for image {img_path}. Clipping."
                )
                p_ai = min(max(p_ai, 0.0), 1.0)

            P_ai[i, j] = p_ai
            X[i, j] = logit(p_ai) if use_logit else p_ai

    if not np.isfinite(X).all():
        bad = np.sum(~np.isfinite(X))
        print(f"[WARN] Feature matrix still contains {bad} non-finite values. Replacing them with 0.0.")
        X[~np.isfinite(X)] = 0.0

    return X, P_ai


def maybe_add_is_face_feature(X: np.ndarray, face_flags: np.ndarray) -> np.ndarray:
    return np.concatenate([X, face_flags.reshape(-1, 1).astype(np.float32)], axis=1)


def _safe_auc(y_true, y_score) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def eval_meta_masked(split_name: str, clf, X: np.ndarray, y: np.ndarray, mask: np.ndarray):
    if mask.sum() == 0:
        print(f"[WARN] META {split_name}: subset is empty.")
        return
    Xs = X[mask]
    ys = y[mask]
    proba_ai = clf.predict_proba(Xs)[:, list(clf.classes_).index(LABEL_MAP["ai"])]
    pred = (proba_ai >= 0.5).astype(np.int32)
    acc = accuracy_score(ys, pred)
    auc = _safe_auc(ys, proba_ai)
    print(f"[RESULT] META {split_name}: ACC={acc:.3f}, AUC={auc:.3f}, n={len(ys)}")


def eval_base_methods_masked(
    split_name: str,
    P_ai: np.ndarray,
    y: np.ndarray,
    method_names: List[str],
    mask: np.ndarray,
    only_methods: Optional[Set[str]] = None,
):
    if mask.sum() == 0:
        print(f"[WARN] Base methods {split_name}: subset is empty.")
        return

    Ps = P_ai[mask]
    ys = y[mask]

    for j, method_name in enumerate(method_names):
        if only_methods is not None and method_name not in only_methods:
            continue
        if only_methods is None and method_name in FACE_ONLY_MODEL_NAMES:
            continue

        proba_ai = Ps[:, j]
        pred = (proba_ai >= 0.5).astype(np.int32)
        acc = accuracy_score(ys, pred)
        auc = _safe_auc(ys, proba_ai)
        print(f"  - {method_name}: ACC={acc:.3f}, AUC={auc:.3f}, n={len(ys)}")


def eval_report(
    split_label: str,
    clf,
    X_meta: np.ndarray,
    y: np.ndarray,
    P_ai: np.ndarray,
    method_names: List[str],
    face_flags: np.ndarray,
):
    mask_face = face_flags == 1
    mask_noface = face_flags == 0
    mask_all = np.ones_like(face_flags, dtype=bool)

    print(f"\n=== REPORT: {split_label} ===")

    eval_meta_masked(f"{split_label} (FACE)", clf, X_meta, y, mask_face)
    eval_meta_masked(f"{split_label} (NOFACE)", clf, X_meta, y, mask_noface)
    eval_meta_masked(f"{split_label} (ALL)", clf, X_meta, y, mask_all)

    print(f"\n[RESULT] Base methods (non-face-only) on {split_label}:")
    print("  (FACE)")
    eval_base_methods_masked(f"{split_label} (FACE)", P_ai, y, method_names, mask_face, only_methods=None)
    print("  (NOFACE)")
    eval_base_methods_masked(f"{split_label} (NOFACE)", P_ai, y, method_names, mask_noface, only_methods=None)
    print("  (ALL)")
    eval_base_methods_masked(f"{split_label} (ALL)", P_ai, y, method_names, mask_all, only_methods=None)

    if len(FACE_ONLY_MODEL_NAMES) > 0:
        print(f"\n[RESULT] Face-only methods on {split_label} (FACE only):")
        eval_base_methods_masked(
            f"{split_label} (FACE)",
            P_ai,
            y,
            method_names,
            mask_face,
            only_methods=FACE_ONLY_MODEL_NAMES,
        )
    else:
        print("\n[INFO] FACE_ONLY_MODEL_NAMES is empty -> no face-only methods to report.")


def create_catboost_meta_model():
    try:
        gpu_count = get_gpu_device_count()
    except Exception:
        gpu_count = 0

    if gpu_count > 0:
        task_type = "GPU"
        print(f"[INFO] Detected {gpu_count} GPU device(s). Using GPU for CatBoost.")
        clf = CatBoostClassifier(
            task_type="GPU",
            devices="0",
            iterations=1000,
            depth=6,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
        )
    else:
        task_type = "CPU"
        print("[INFO] No GPU devices detected. Using CPU for CatBoost.")
        clf = CatBoostClassifier(
            task_type="CPU",
            iterations=1000,
            depth=6,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
        )

    print(f"[INFO] CatBoost task_type = {task_type}")
    return clf, task_type


def main():
    method_names = [m.name for m in BASE_MODELS]
    print("[INFO] Base methods:", method_names)
    print(f"[INFO] ROOT_DIR = {ROOT_DIR}")
    print(f"[INFO] FACES_DB_PATH = {FACES_DB_PATH}")
    print(f"[INFO] USE_LOGIT = {USE_LOGIT}")
    print(f"[INFO] ADD_IS_FACE_FEATURE = {ADD_IS_FACE_FEATURE}")
    print(f"[INFO] LABEL_MAP = {LABEL_MAP}  (AI=1, REAL=0)")
    print("[INFO] Feature semantics:", "logit(p_ai)" if USE_LOGIT else "p_ai")

    face_only_mask = np.array([name in FACE_ONLY_MODEL_NAMES for name in method_names], dtype=bool)
    if face_only_mask.any():
        face_only_list = [n for n in method_names if n in FACE_ONLY_MODEL_NAMES]
        print(f"[INFO] Face-only methods enabled ({len(face_only_list)}): {face_only_list}")
    else:
        print("[INFO] No face-only methods configured.")

    train_paths, train_labels = collect_image_paths(ROOT_DIR, split="train")
    train_face_flags = load_face_flags(FACES_DB_PATH, ROOT_DIR, train_paths, default_if_missing=0)
    print(f"[INFO] TRAIN: faces={int((train_face_flags==1).sum())}, nofaces={int((train_face_flags==0).sum())}")

    X_train, P_train = compute_features_for_images(
        train_paths,
        BASE_MODELS,
        face_flags=train_face_flags,
        face_only_mask=face_only_mask,
        use_logit=USE_LOGIT,
        neutral_p=NEUTRAL_P_AI_FOR_SKIPPED,
    )
    if ADD_IS_FACE_FEATURE:
        X_train_meta = maybe_add_is_face_feature(X_train, train_face_flags)
    else:
        X_train_meta = X_train

    val_paths, val_labels = collect_image_paths(ROOT_DIR, split="val")
    val_face_flags = load_face_flags(FACES_DB_PATH, ROOT_DIR, val_paths, default_if_missing=0)
    print(f"[INFO] VAL: faces={int((val_face_flags==1).sum())}, nofaces={int((val_face_flags==0).sum())}")

    X_val, P_val = compute_features_for_images(
        val_paths,
        BASE_MODELS,
        face_flags=val_face_flags,
        face_only_mask=face_only_mask,
        use_logit=USE_LOGIT,
        neutral_p=NEUTRAL_P_AI_FOR_SKIPPED,
    )
    if ADD_IS_FACE_FEATURE:
        X_val_meta = maybe_add_is_face_feature(X_val, val_face_flags)
    else:
        X_val_meta = X_val

    clf, task_type = create_catboost_meta_model()
    print("[INFO] Training meta-model (CatBoost)...")
    clf.fit(X_train_meta, train_labels, eval_set=(X_val_meta, val_labels))

    eval_report("TRAIN", clf, X_train_meta, train_labels, P_train, method_names, train_face_flags)
    eval_report("VAL", clf, X_val_meta, val_labels, P_val, method_names, val_face_flags)

    meta_package = {
        "clf": clf,
        "model_names": method_names,
        "use_logit": USE_LOGIT,
        "meta_model_type": "CatBoostClassifier",
        "task_type": task_type,
        "label_map": LABEL_MAP,
        "feature_semantics": "logit_p_ai" if USE_LOGIT else "p_ai",
        "faces_db_path": FACES_DB_PATH,
        "face_only_model_names": sorted(list(FACE_ONLY_MODEL_NAMES)),
        "neutral_p_ai_for_skipped_face_models": float(NEUTRAL_P_AI_FOR_SKIPPED),
        "add_is_face_feature": bool(ADD_IS_FACE_FEATURE),
        "note": "VAL contains former TEST (merged). No separate test split used.",
    }
    dump(meta_package, OUTPUT_PATH)
    print(f"[INFO] Meta-model saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
