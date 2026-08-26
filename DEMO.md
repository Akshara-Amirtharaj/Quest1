# Quest1 demo

The demo is a thin local UI and HTTP adapter around `dialogue_locator.pipeline.run_v2`.
It does not implement or alter localization logic.

## API

From the repository root, install the application and demo adapter together:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -U pip setuptools wheel
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\quest1-api.exe
```

The API listens on `http://127.0.0.1:8000` and writes runtime artifacts beneath
`.cache/demo` by default. Set `QUEST1_DEMO_DATA_DIR` to use another location.

The same FastAPI process serves the frontend. Open `http://127.0.0.1:8000`.
Do not add `--reload` for submission/demo runs: long model and cache activity must
not restart an in-flight request.
