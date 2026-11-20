"""Put the course datasets where the notes' cells look for them: workspace/data/ (the live kernel's working folder).

    python tools/setup_data.py

Copies what the repository already has (Fashion-MNIST, the Telco churn CSV, pizza/steak/sushi images); nothing is
downloaded. workspace/data/ is not in git. Safe to run again: existing files are left alone.
"""
import pathlib, shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "workspace" / "data"
SOURCES = {
    "FashionMNIST": ROOT / "cnn" / "data" / "FashionMNIST",
    "Telco-Customer-Churn.csv": ROOT / "customer-churn-prediction" / "data" / "Telco-Customer-Churn.csv",
    "pizza_steak_sushi": ROOT / "transfer-learning" / "data" / "pizza_steak_sushi",
}
DATA.mkdir(parents=True, exist_ok=True)
for name, src in SOURCES.items():
    dst = DATA / name
    if dst.exists():
        print(f"ok       {name}")
    elif not src.exists():
        print(f"MISSING  {name}: expected at {src}")
    elif src.is_dir():
        shutil.copytree(src, dst); print(f"copied   {name}/")
    else:
        shutil.copy2(src, dst); print(f"copied   {name}")
