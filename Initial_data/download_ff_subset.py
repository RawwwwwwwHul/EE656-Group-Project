from kaggle.api.kaggle_api_extended import KaggleApi
import random
import os
import re
import zipfile

DATASET = "xdxd003/ff-c23"
OUTPUT = "mini_ffpp"

random.seed(42)

api = KaggleApi()
api.authenticate()

# ----------------------------
# Get ALL files (pagination)
# ----------------------------
all_files = []
page_token = None

print("Fetching file list...")

while True:

    res = api.dataset_list_files(
        DATASET,
        page_token=page_token,
        page_size=1000
    )
    all_files.extend(res.files)

    print(f"Collected {len(all_files)} files")

    page_token = getattr(res, "_next_page_token", None)

    if page_token is None:
        break

print(f"\nTotal files: {len(all_files)}")

# ----------------------------
# Categorize
# ----------------------------
cats = {
    "original": [],
    "DeepFakeDetection": [],
    "Face2Face": [],
    "FaceSwap": [],
    "NeuralTextures": []
}

for f in all_files:

    name = f.name

    if "/original/" in name.lower():
        cats["original"].append(name)

    elif "/DeepFakeDetection/" in name:
        cats["DeepFakeDetection"].append(name)

    elif "/Face2Face/" in name:
        cats["Face2Face"].append(name)

    elif "/FaceSwap/" in name:
        cats["FaceSwap"].append(name)

    elif "/NeuralTextures/" in name:
        cats["NeuralTextures"].append(name)

print("\nCategory counts")
for k in cats:
    print(k, len(cats[k]))

# ----------------------------
# Sample fake videos
# ----------------------------
chosen = {}
needed_originals = set()

for cat in ["DeepFakeDetection","Face2Face","FaceSwap","NeuralTextures"]:

    chosen[cat] = random.sample(cats[cat],5)

    for file in chosen[cat]:

        fname = os.path.basename(file)

        m = re.match(r"(\d+)_(\d+)__",fname)

        if m:
            needed_originals.add(m.group(1))
            needed_originals.add(m.group(2))

# ----------------------------
# Match originals
# ----------------------------
lookup = {}

for file in cats["original"]:

    vid = os.path.splitext(os.path.basename(file))[0]

    lookup[vid] = file

chosen_originals = []

for vid in needed_originals:

    if vid in lookup:
        chosen_originals.append(lookup[vid])

remaining = list(set(cats["original"]) - set(chosen_originals))

if len(chosen_originals) < 100:
    chosen_originals.extend(
        random.sample(
            remaining,
            100-len(chosen_originals)
        )
    )

chosen["original"] = chosen_originals[:20]

print("\nDownload Summary")
for k in chosen:
    print(k,len(chosen[k]))

# ----------------------------
# Download
# ----------------------------
total = sum(len(v) for v in chosen.values())
done = 0

for cat in chosen:

    folder = os.path.join(OUTPUT,cat)
    os.makedirs(folder,exist_ok=True)

    for file in chosen[cat]:

        done += 1

        print(f"[{done}/{total}] {file}")

        api.dataset_download_file(
            DATASET,
            file_name=file,
            path=folder,
            force=False,
            quiet=True
        )

        # unzip automatically
        zip_path = os.path.join(folder, os.path.basename(file)+".zip")

        if os.path.exists(zip_path):

            with zipfile.ZipFile(zip_path,'r') as z:
                z.extractall(folder)

            os.remove(zip_path)

print("\nFinished!")