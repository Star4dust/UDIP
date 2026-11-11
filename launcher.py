import subprocess, sys, os, webbrowser, time, urllib.request
from pathlib import Path

def server_ready():
    try:
        urllib.request.urlopen('http://localhost:8501', timeout=1)
        return True
    except:
        return False

def find_app_dir():
    # Read path written by Build_EXE.bat
    candidates = [
        Path(sys.executable).parent / "_udip_path.txt",
        Path(__file__).parent / "_udip_path.txt",
    ]
    for c in candidates:
        if c.exists():
            p = Path(c.read_text().strip())
            if (p / "app.py").exists():
                return p
    # Fallback: exe is in app root
    for p in [Path(sys.executable).parent, Path(sys.executable).parent.parent]:
        if (p / "app.py").exists():
            return p
    return None

def main():
    app_dir = find_app_dir()
    if not app_dir:
        import tkinter.messagebox as mb
        mb.showerror("UDIP Error", "Cannot find app folder.\nMake sure UDIP.exe is inside the udip_v3 folder.")
        return

    os.chdir(str(app_dir))
    streamlit = app_dir / "venv" / "Scripts" / "streamlit.exe"
    app_py    = app_dir / "app.py"

    if not streamlit.exists():
        import tkinter.messagebox as mb
        mb.showerror("UDIP Error", f"Streamlit not found at:\n{streamlit}\n\nRun Install_UDIP.bat first.")
        return

    p = subprocess.Popen(
        [str(streamlit), "run", str(app_py),
         "--server.port", "8501",
         "--server.headless", "true",
         "--browser.gatherUsageStats", "false",
         "--logger.level", "error"],
        cwd=str(app_dir),
        creationflags=subprocess.CREATE_NO_WINDOW
    )

    for i in range(30):
        if server_ready():
            break
        time.sleep(1)

    webbrowser.open('http://localhost:8501')
    p.wait()

if __name__ == '__main__':
    main()
