# UDIP: Universal Data Integration Platform

A local desktop data analysis application for Windows, built with Python and Streamlit. It lets non-programmers load, clean, merge, query and visualise datasets, with optional AI assistance. It was developed during an internship on the HIRAX radio telescope project and includes tools for radio astronomy data (HDF5 visibility data, calibration, RFI diagnostics).

> Status: working prototype (v3). Not production software. See "Known limitations".

## Features
- **Data upload:** CSV, Excel, JSON, Parquet, HDF5 and more
- **Data Forge:** cleaning, outlier handling and validation rules
- **Merge Wizard:** multi-step dataset joins
- **SQL Editor:** query loaded tables with standard SQL (DuckDB)
- **Visualisation Studio:** interactive Plotly charts
- **Dashboard, scheduler and report generation**
- **AI Assistant (optional):** conversational analysis and code generation through the Groq API
- **Radio astronomy tools:** HIRAX-style data inspection, redundant calibration, RFI and UV-coverage diagnostics
- **Desktop integration:** system tray, launcher and PyInstaller build script (`installer/Build_EXE.bat`)

## Tech stack
Python, Streamlit, DuckDB, pandas, Polars, scikit-learn, Plotly, h5py, Groq API, PyInstaller

## Getting started
Requires Windows 10/11 and Python 3.10 or newer.

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Or run `Run_UDIP.bat`, which sets up the environment on first launch.

### AI assistant (optional)
Create a free API key at https://console.groq.com, then either:
- copy `.env.example` to `.env` and set `GROQ_API_KEY`, or
- paste the key into the AI Assistant page inside the app.

Never commit your key. `.env` and `udip_config.json` are git-ignored.

### Google Drive import (optional)
See `config/google/README.md`.

## Tests
```
python -m pytest tests/
```

## Project layout
```
app.py            Streamlit entry point and page router
launcher.py       Desktop launcher
tray.py           System tray integration
modules/          Backend logic (data, AI, calibration, SQL, state, ...)
pages/, udip_pages/  Streamlit page UIs
plugins/          Plugin extensions
tests/            Test suite
validation_corpus/  Small synthetic regression datasets
docs/             Architecture notes
installer/        Windows installer scripts
```
## How this was built
UDIP was developed with AI coding assistants. I defined the requirements and architecture, then tested and debugged it.

## Known limitations
- Windows-focused: the installer, launcher and tray integration assume Windows.
- When the AI features are used, prompts, and for some features dataset context, are sent to the Groq API. Don't use them on confidential data.
- The AI assistant can generate and run Python code on your machine. Only use it on data and prompts you trust.
- Large datasets can exhaust memory; use sampling for big files.
- The codebase grew quickly and has overlapping modules that need consolidating.
- Test coverage is partial.
