# IFT6758 Group Repo Template

This repository contains the code and assets for the IFT6758 NHL Play-by-Play Analysis project.  
It includes both the main analytical notebook (`milestone1.ipynb`) and a Jekyll-based blog (`myblog`) used to showcase results.

The project demonstrates key data science concepts such as:
- Extracting and processing NHL play-by-play data  
- Visualizing hockey rink and game statistics  
- Building reproducible environments for collaboration and deployment  
<p align="center">
<img src="./figures/nhl_rink.png" alt="NHL Rink is 200ft x 85ft." width="400"/>
<p>

Also included in this repo is an image of the NHL ice rink that you can use in your plots.
It has the correct location of lines, faceoff dots, and length/width ratio as the real NHL rink.
Note that the rink is 200 feet long and 85 feet wide, with the goal line 11 feet from the nearest edge of the rink, and the blue line 75 feet from the nearest edge of the rink.

The image can be found in [`./figures/nhl_rink.png`](./figures/nhl_rink.png).

**Directory overview:**
Each folder serves a specific purpose:

- **`ift6758/`**: Main package directory containing source files, notebooks, and blog content  
  - **`src/`**: Core working directory for scripts, data, and Jupyter notebooks  
  - **`myblog/`**: Contains `_posts/`, configuration files, and assets for the Jekyll site  
  - **`_site/`**: Automatically generated output folder created by the Jekyll build process  

- **`figures/`**: Stores figures and static images for reports or README use  

- **`notebooks/`**: Optional folder for scratch or exploratory notebooks  

- **`README.md`** — This file
- **`environments.yml`**
- **`gitignore.txt`**
- **`requirements.txt`**


## Set up Instructions

Follow these steps to reproduce the environment and run both the notebook and the Jekyll blog locally. 

    git clone <YOUR_REPOSITORY_URL>
    cd <YOUR_REPOSITORY_FOLDER>

## Downloading Source Folder and Unzipping Files

Download the source folder and all its objects into VS Code or whatever software you are using. 
If you have received a compressed version of the milestone

unzip milestone1.zip
mv milestone1.ipynb src/

This ensures the notebook is inside the src directory where all source code and data live.

## Create and Activate a virtual Environment

We’ll use venv for simplicity. From inside the src directory:

python -m venv venv
source venv/bin/activate     # On macOS/Linux
venv\Scripts\activate        # On Windows

## Install Dependencies

pip install -r requirements.txt

This will install all Python dependencies required for the notebook and analysis.

### Run the milestone notebook

Launch Jupyter Lab or Notebook and open milestone1.ipynb:

jupyter lab

Then execute all cells to reproduce the results and visualizations.

### Pip + Virtualenv

An alternative to Conda is to use pip and virtualenv to manage your environments.
This may play less nicely with Windows, but works fine on Unix devices.
This method makes use of the `requirements.txt` file; you can disregard the `environment.yml` file if you choose this method.

Ensure you have installed the [virtualenv tool](https://virtualenv.pypa.io/en/latest/installation.html) on your system.
Once installed, create a new virtual environment:

    vitualenv ~/ift6758-venv
    source ~/ift6758-venv/bin/activate

Install the packages from a requirements.txt file:

    pip install -r requirements.txt

As before, register the environment so jupyter can see it:

    python -m ipykernel install --user --name=ift6758-venv

You should now be able to launch jupyter and see your conda environment:

    jupyter-lab

If you want to create a new `requirements.txt` file, you can use `pip freeze`:

    pip freeze > requirements.txt
