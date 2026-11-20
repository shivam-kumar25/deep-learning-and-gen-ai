"""Tile a subject's figure outputs into one image for a quick visual check.  Usage: contact_sheet.py <images/out dir> <out.png> [prefix ...]"""
import glob, math, os, sys
import matplotlib; matplotlib.use("agg")
import matplotlib.pyplot as plt, matplotlib.image as mpimg
d, out, pre = sys.argv[1], sys.argv[2], sys.argv[3:]
fs = [f for f in sorted(glob.glob(os.path.join(d, "*.png"))) if not pre or any(os.path.basename(f).startswith(p) for p in pre)]
cols = 3; rows = math.ceil(len(fs) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 3.4))
for ax in axes.ravel(): ax.axis("off")
for ax, f in zip(axes.ravel(), fs):
    ax.imshow(mpimg.imread(f)); ax.set_title(os.path.basename(f), fontsize=9)
plt.tight_layout(); fig.savefig(out, dpi=70); print(len(fs), "figures ->", out)
