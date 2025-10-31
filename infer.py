import os
import json
import argparse
import pickle
import numpy as np
import pandas as pd

from model import Ensemble  # reuse your concatenation logic

MODELS_DIR = "models"

def _vid_name(name: str) -> str:
    # same logic used in model.py training block
    return name.replace(name.split('_')[-1], '')[0:-1]

def load_models():
    # meta
    with open(os.path.join(MODELS_DIR, "meta.json"), "r") as f:
        meta = json.load(f)
    regions = meta["regions"]
    num_frames = int(meta["num_frames"])

    # classifier
    with open(os.path.join(MODELS_DIR, "classifier.pkl"), "rb") as f:
        classifier = pickle.load(f)

    # per-region DefakeHop
    defakehops = {}
    for r in regions:
        with open(os.path.join(MODELS_DIR, f"defakehop_{r}.pkl"), "rb") as f:
            defakehops[r] = pickle.load(f)

    return regions, num_frames, classifier, defakehops

def run_inference(npz_dir: str, split: str):
    regions, num_frames, classifier, defakehops = load_models()

    # build an Ensemble shell to reuse predict concatenation
    ens = Ensemble(regions=regions, num_frames=num_frames, verbose=True)
    ens.classifier = classifier
    ens.defakeHop = defakehops

    # feed each region's test npz
    for region in regions:
        path = os.path.join(npz_dir, f"{region}.{split}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing NPZ for region '{region}': {path}")
        data = np.load(path)
        images = data["images"]
        names = data["names"]
        ens.predict_region(region, images, names)

    # frame-level prediction
    frame_prob, frame_names = ens.predict_classifier(clean=False)

    # aggregate to video-level (mean of frames per video)
    vids = {}
    for p, n in zip(frame_prob, frame_names):
        vn = _vid_name(n)
        if vn not in vids:
            vids[vn] = []
        vids[vn].append(float(p))
    video_rows = []
    for vn, arr in vids.items():
        video_rows.append({"name": vn, "prob": float(np.mean(arr))})

    df = pd.DataFrame(video_rows).sort_values("name")
    out_csv = "inference_output.csv"
    df.to_csv(out_csv, index=False)

    # console summary
    print("\n=============== Video-level Results ===============")
    for _, row in df.iterrows():
        print(f"{row['name']}, prob={row['prob']:.6f}  -->  {'FAKE' if row['prob']>=0.5 else 'REAL'}")
    print(f"\nSaved: {out_csv}")

def main():
    ap = argparse.ArgumentParser(description="Run video-level inference using saved DefakeHop models.")
    ap.add_argument("--npz_dir", default="data", help="Folder containing <region>.<split>.npz files")
    ap.add_argument("--split", default="test", help="Split suffix used in NPZ filenames (default: test)")
    args = ap.parse_args()
    run_inference(args.npz_dir, args.split)

if __name__ == "__main__":
    main()
