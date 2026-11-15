# NASA Agri Intelligence

This project turns NASA Earth observation data into a task-based agricultural analysis workflow. It pulls AppEEARS/NASA datasets for a selected area, standardizes the downloaded rasters, summarizes them by AOI grid cells, and exposes the result in a local web interface for inspection and downstream analysis.

## How it works

1. Define a study area and collection settings.
2. Submit a NASA/AppEEARS task for the chosen product and layers.
3. Download and validate the task output.
4. Standardize the rasters using NASA metadata, masks, and QA rules.
5. Aggregate the data into AOI grid cells for spatial analysis.
6. View and manage the workflow through the local site and API.

The app is built around a Python backend plus a small frontend. The backend handles NASA task management, downloads, data validation, and processing; the frontend gives a quick site for interacting with tasks and results locally.

## Quickstart the site

Requirements:
- Python 3.12+
- Node.js + npm

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,web]"
npm --prefix frontend install
```

Create your environment file and enter your Earthdata username and password:

```bash
cp .env.example .env
```

Generate the AppEEARS token from those credentials:

```bash
python get_token.py
```

This saves `NASA_APP_EARS_TOKEN` to `.env`. Run the command from the project root.

Start the backend:

```bash
source .venv/bin/activate
python -m uvicorn nasa_agri_intelligence.web:app --host 127.0.0.1 --port 8000
```

Start the frontend in another terminal:

```bash
npm --prefix frontend run dev
```

Open the frontend in your browser at:

```text
http://127.0.0.1:5173
```

Optional sanity check before using real NASA tasks:

```bash
source .venv/bin/activate
python main.py validate
```

Then submit a task and work through the site flow as needed:

```bash
python main.py tasks submit --product MOD13Q1.061 --task-name MY_TASK
python main.py tasks list
```

This repo is meant for local workflow use and development; the site is intended to run on localhost and is not a production public deployment.
