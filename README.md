# IFT6758 Group Project — NHL Play-by-Play Analysis

This repository contains the code and assets for the IFT6758 NHL Play-by-Play Analysis project.  
It includes both the main analytical notebook (`milestone1.ipynb`) and a Jekyll-based blog (`myblog`) used to showcase results.

The project demonstrates key data science concepts such as:
- Extracting and processing NHL play-by-play data  
- Visualizing hockey rink and game statistics  
- Building reproducible environments for collaboration and deployment  

---

## Repository Structure

**Directory overview:**

- **src/**
  - `milestone1.ipynb` — Main analysis notebook  
  - `nhl_all_games_data.csv` — Dataset used for analysis  
  - `rink.png` — NHL rink image for visualizations  
  - `requirements.txt` — Python dependencies  

- **myblog/**
  - `_posts/` — Blog posts (Markdown)  
  - `assets/images/` — Blog images  
  - `_config.yml` — Jekyll site configuration  
  - `index.markdown` — Blog homepage  
  - `Gemfile`, `Gemfile.lock` — Ruby dependencies for Jekyll  

- **README.md** — This file  

---

## Setup Instructions

Follow these steps to reproduce the environment and run both the notebook and the Jekyll blog locally.

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>

### 2. Unzip milestone1 code fle

unzip milestone1.zip
mv milestone1.ipynb src/
