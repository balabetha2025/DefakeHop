import os
import json
import pickle
import numpy as np

from model import Ensemble

MODELS_DIR = "models"

def detect_regions(data_dir="data"):
    cand = ["left_eye", "right_eye", "mouth"]
    have = []
    for r in cand:
        if os.path.exists(os.path.join(data_dir, f"{r}.train.npz")):
            have.append(r)
    if not have:
        raise FileNotFoundError("No <region>.train.npz files found in ./data")
    return have

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 👇 auto-detect what you actually have in ./data
    regions = detect_regions("data")
    num_frames = 6
    multi_cwSaab_parm = dict(
        num_hop=3, kernel_sizes=[3,3,3], split_thr=0.01, keep_thr=0.001,
        max_channels=[10,10,10], spatial_components=[0.9,0.9,0.9], n_jobs=4, verbose=True
    )

    model = Ensemble(regions=regions, num_frames=num_frames, verbose=True)

    # train per region
    for region in model.regions:
        path = f'data/{region}.train.npz'
        data = np.load(path)
        train_images = data['images']
        train_labels = data['labels']
        train_names = data['names']
        print(train_images.shape)
        model.fit_region(region, train_images, train_labels, train_names, multi_cwSaab_parm)

    # train top classifier (don’t clean so we can save defakeHop objects)
    train_prob, train_vid_names = model.train_classifier(clean=False)

    # ---- SAVE ----
    with open(os.path.join(MODELS_DIR, "meta.json"), "w") as f:
        json.dump({"regions": regions, "num_frames": num_frames}, f, indent=2)

    with open(os.path.join(MODELS_DIR, "classifier.pkl"), "wb") as f:
        pickle.dump(model.classifier, f)

    for region, dfh in model.defakeHop.items():
        with open(os.path.join(MODELS_DIR, f"defakehop_{region}.pkl"), "wb") as f:
            pickle.dump(dfh, f)

    print("\nSaved:")
    print(f"- {MODELS_DIR}/meta.json")
    print(f"- {MODELS_DIR}/classifier.pkl")
    for r in regions:
        print(f"- {MODELS_DIR}/defakehop_{r}.pkl")

if __name__ == "__main__":
    main()
