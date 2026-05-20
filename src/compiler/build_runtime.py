import os
import glob

src_dir = os.path.join(os.path.dirname(__file__), "runtime_modules")
out_file = os.path.join(os.path.dirname(__file__), "ml_runtime.py")

modules = sorted(glob.glob(os.path.join(src_dir, "*.py")))

with open(out_file, "w", encoding="utf-8") as out:
    for mod in modules:
        with open(mod, "r", encoding="utf-8") as f:
            out.write(f.read())

print(f"Successfully generated ml_runtime.py from {len(modules)} modules.")
