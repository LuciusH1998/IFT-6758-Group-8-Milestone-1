import nbformat

path = "milestone1.ipynb"
nb = nbformat.read(path, as_version=4)

# Ensure widgets metadata exists and is valid
if "widgets" not in nb.metadata:
    nb.metadata["widgets"] = {}

if "state" not in nb.metadata["widgets"]:
    nb.metadata["widgets"]["state"] = {}

# Add version info to satisfy GitHub nbviewer
nb.metadata["widgets"]["version_major"] = 2
nb.metadata["widgets"]["version_minor"] = 0

# Save notebook
nbformat.write(nb, path)
print("✅ Notebook metadata repaired and saved!")
