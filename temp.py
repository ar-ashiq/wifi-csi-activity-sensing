import os
from pathlib import Path

base = Path(r"D:\Ub_Hack\hack_2")
print("Folders and files inside D:\\Ub_Hack\\hack_2:")
for p in base.iterdir():
    print(f" - {p.name} ({'DIR' if p.is_dir() else 'FILE'})")