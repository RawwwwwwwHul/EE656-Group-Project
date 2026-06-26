import os
import random
import shutil

random.seed(42)      # Reproducible selection

FAKE_DIR = r"D:\Summer\ISTVT\clips\fake"
BACKUP_DIR = r"D:\Summer\ISTVT\clips\fake_unused"

os.makedirs(BACKUP_DIR, exist_ok=True)

clips = [
    f for f in os.listdir(FAKE_DIR)
    if os.path.isfile(os.path.join(FAKE_DIR, f))
]

print(f"Total fake clips: {len(clips)}")

# Keep 6000 randomly
keep = set(random.sample(clips, 6000))

moved = 0

for clip in clips:
    if clip not in keep:
        shutil.move(
            os.path.join(FAKE_DIR, clip),
            os.path.join(BACKUP_DIR, clip)
        )
        moved += 1

print(f"Kept: {len(keep)}")
print(f"Moved: {moved}")