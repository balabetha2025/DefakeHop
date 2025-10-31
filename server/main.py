# server/main.py
import io, os, json, tempfile, shutil, subprocess
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, File, UploadFile, HTTPException, Security
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

import numpy as np
from PIL import Image
import uvicorn

# --- your project imports ---
import utils
from model import Ensemble
import imageio.v2 as imageio

# =========================
# CONFIG
# =========================
API_KEY = os.getenv("API_KEY", "devkey")        # set via env in Docker/run
MODELS_DIR = Path("models")
FPS = 6                                         # frames per second for sampling
THRESHOLD = 0.5                                 # label cutoff

# =========================
# APP & SECURITY (shows padlock in Swagger)
# =========================
security = HTTPBearer(auto_error=False)

app = FastAPI(
    title="DefakeHop API",
    version="1.0.0",
    description="Upload a video and get a probability of being fake. "
                "Use the 🔒 Authorize button (top-right) with your API key."
)

def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)) -> None:
    """
    Validates Bearer token. Using Security() makes Swagger show the padlock.
    """
    if not API_KEY:
        return  # auth disabled if no key is set
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Bad token")

# =========================
# LOAD MODELS AT STARTUP
# =========================
def load_artifacts() -> Ensemble:
    meta_fp = MODELS_DIR / "meta.json"
    clf_fp  = MODELS_DIR / "classifier.pkl"
    if not meta_fp.exists() or not clf_fp.exists():
        raise RuntimeError("Model artifacts missing in ./models. Did you `git lfs pull`?")

    with open(meta_fp) as f:
        meta = json.load(f)
    regions: List[str] = meta["regions"]
    num_frames: int = meta["num_frames"]

    ens = Ensemble(regions=regions, num_frames=num_frames, verbose=False)

    # Load top classifier
    import pickle
    with open(clf_fp, "rb") as f:
        ens.classifier = pickle.load(f)

    # Load DefakeHop per region
    for r in regions:
        dfh_fp = MODELS_DIR / f"defakehop_{r}.pkl"
        if not dfh_fp.exists():
            raise RuntimeError(f"Missing {dfh_fp}")
        with open(dfh_fp, "rb") as f:
            ens.defakeHop[r] = pickle.load(f)

    return ens

INFER: Optional[Ensemble] = None

@app.on_event("startup")
def _startup() -> None:
    global INFER
    INFER = load_artifacts()

# =========================
# HELPERS
# =========================
def _run(cmd: list) -> None:
    """Run a subprocess; raise HTTP 400 with stderr if it fails."""
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"Command failed: {' '.join(cmd)} :: {e}")

def _extract_frames(video_fp: str, out_dir: str, fps: int = FPS) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
        "-i", video_fp,
        "-vf", f"fps={fps}",
        f"{out_dir}/frame_%06d.jpg"
    ]
    _run(cmd)

# --- quick landmark + patches using face_alignment (CPU) ---
def _face_landmarks(img_np: np.ndarray):
    import face_alignment
    from face_alignment import LandmarksType
    LM_TYPE = getattr(LandmarksType, "TWO_D", None) or getattr(LandmarksType, "_2D")
    fa = face_alignment.FaceAlignment(LM_TYPE, device="cpu", flip_input=False, face_detector="sfd")
    if hasattr(fa.face_detector, "filter_threshold"):
        fa.face_detector.filter_threshold = 0.2
    lms = fa.get_landmarks(img_np)
    if not lms:
        return None
    # choose the biggest face (area)
    def area(pts):
        pts = np.asarray(pts)
        x0, y0 = np.min(pts, axis=0); x1, y1 = np.max(pts, axis=0)
        return max(0, x1-x0) * max(0, y1-y0)
    return max(lms, key=area)

# indices for left/right eye rough boxes from 68-pt scheme
LEFT_EYE_IDX  = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 48))
MOUTH_IDX     = list(range(48, 68))

def _crop_from_landmarks(img_np: np.ndarray, lm, idxs, box_pad: int = 8):
    pts = np.asarray([lm[i] for i in idxs])
    x0, y0 = np.min(pts, axis=0).astype(int)
    x1, y1 = np.max(pts, axis=0).astype(int)
    x0 = max(x0 - box_pad, 0); y0 = max(y0 - box_pad, 0)
    x1 = min(x1 + box_pad, img_np.shape[1]-1); y1 = min(y1 + box_pad, img_np.shape[0]-1)
    crop = img_np[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    pil = Image.fromarray(crop).resize((32, 32))
    return np.array(pil)

def _build_npz_like(frames_dir: Path, vid_id: str, regions: list) -> Dict[str, Dict[str, np.ndarray]]:
    images = {r: [] for r in regions}
    labels = {r: [] for r in regions}
    names  = {r: [] for r in regions}

    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        return {r: dict(images=np.array([]), labels=np.array([]), names=np.array([])) for r in regions}

    for i, fp in enumerate(frame_files):
        img = imageio.imread(fp)
        lm = _face_landmarks(img)
        if lm is None:
            continue
        if "left_eye" in regions:
            c = _crop_from_landmarks(img, lm, LEFT_EYE_IDX)
            if c is not None:
                images["left_eye"].append(c); labels["left_eye"].append(0); names["left_eye"].append(f"real/{vid_id}_{i:06d}.jpg")
        if "right_eye" in regions:
            c = _crop_from_landmarks(img, lm, RIGHT_EYE_IDX)
            if c is not None:
                images["right_eye"].append(c); labels["right_eye"].append(0); names["right_eye"].append(f"real/{vid_id}_{i:06d}.jpg")
        if "mouth" in regions:
            c = _crop_from_landmarks(img, lm, MOUTH_IDX)
            if c is not None:
                images["mouth"].append(c); labels["mouth"].append(0); names["mouth"].append(f"real/{vid_id}_{i:06d}.jpg")

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for r in regions:
        out[r] = dict(
            images=np.asarray(images[r], dtype=np.uint8),
            labels=np.asarray(labels[r], dtype=int),
            names=np.asarray(names[r], dtype=str),
        )
    return out

def _infer_on_video_bytes(b: bytes):
    tmp_root = Path(tempfile.mkdtemp(prefix="defake-"))
    try:
        vid_fp = tmp_root / "input.mp4"
        with open(vid_fp, "wb") as f:
            f.write(b)

        frames_dir = tmp_root / "frames"
        _extract_frames(str(vid_fp), str(frames_dir), fps=FPS)

        built = _build_npz_like(frames_dir, "upload", INFER.regions)

        INFER.clean_buffer()
        have_any = False
        for r in INFER.regions:
            arr = built[r]["images"]
            nm  = built[r]["names"]
            if arr.size == 0:
                continue
            INFER.predict_region(r, arr, nm)
            have_any = True

        if not have_any:
            return 0.5, "upload"

        probs, names = INFER.predict_classifier(clean=True)
        if len(probs) == 0:
            return 0.5, "upload"

        _, vid_probs, vid_names = utils.vid_prob(probs, names)
        return float(vid_probs[0]), vid_names[0]
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

# =========================
# API
# =========================
class URLIn(BaseModel):
    video_url: str

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/predict")
async def predict(
    creds: Optional[HTTPAuthorizationCredentials] = Security(security),
    video: Optional[UploadFile] = File(default=None)
):
    # Auth
    verify_token(creds)

    if video is None:
        raise HTTPException(status_code=422, detail="Send a file field named 'video'")

    try:
        b = await video.read()
        if not b:
            raise HTTPException(status_code=400, detail="Empty upload")
        prob, name = _infer_on_video_bytes(b)
        label = "fake" if prob >= THRESHOLD else "real"
        return JSONResponse({"success": True, "video": Path(name).stem, "probability_fake": prob, "label": label})
    except HTTPException:
        raise
    
    except Exception as e:
        print("\n\n=== SERVER ERROR ===")
        traceback.print_exc()
        print("=== END ERROR ===\n\n")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

# uvicorn entrypoints
if __name__ == "__main__":
    uvicorn.run("server.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "7860")), reload=False)
