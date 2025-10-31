import os
import multiprocessing
from pathlib import Path
import numpy as np
import imageio.v2 as imageio

def LandmarkExtractor(file_path, output_dir):
    import face_alignment
    from face_alignment import LandmarksType

    img = imageio.imread(file_path)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / (Path(file_path).stem + ".npy")
    if out_fp.exists():
        return True

    LM_TYPE = getattr(LandmarksType, "TWO_D", None) or getattr(LandmarksType, "_2D")
    fa = face_alignment.FaceAlignment(
        LM_TYPE,
        device="cpu",
        flip_input=False,
        face_detector="sfd"
    )
    if hasattr(fa.face_detector, "filter_threshold"):
        fa.face_detector.filter_threshold = 0.2

    lms = fa.get_landmarks(img)
    if not lms:
        print(f"[NO_FACE] {file_path}")
        return False

    def area(pts):
        pts = np.asarray(pts)
        x0, y0 = np.min(pts, axis=0)
        x1, y1 = np.max(pts, axis=0)
        return max(0, x1 - x0) * max(0, y1 - y0)

    lm = max(lms, key=area)
    np.save(out_fp, np.asarray(lm, dtype=np.float32))
    return True

if __name__ == "__main__":
    roots = ["data/train/fake", "data/train/real", "data/test/real", "data/test/fake"]
    params = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for fname in os.listdir(root):
            if fname.lower().endswith(".bmp"):
                in_fp = os.path.join(root, fname)
                out_dir = os.path.join("landmarks", root)
                params.append([in_fp, out_dir])

    os.makedirs("landmarks", exist_ok=True)
    pool = multiprocessing.Pool(processes=1)
    pool.starmap(LandmarkExtractor, params)
    pool.close()
    pool.join()
    print("Done.")
