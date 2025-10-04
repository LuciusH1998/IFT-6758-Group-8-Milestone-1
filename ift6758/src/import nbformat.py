import nbformat

# Read notebook
path = "milestone1.ipynb"
nb = nbformat.read(path, as_version=4)

# If widgets metadata missing state, fix it
if "widgets" in nb.metadata and "state" not in nb.metadata["widgets"]:
    nb.metadata["widgets"]["state"] = {}
    print("Added empty 'state' key to metadata.widgets")

# Save notebook
nbformat.write(nb, path)
print("Notebook fixed and saved")
