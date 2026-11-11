"""
UDIP v3 — tray.py
System tray application. Run this instead of Run_UDIP.bat.
- Sits in Windows notification area (bottom-right clock area)
- No black terminal window
- Right-click menu: Open, Restart, Start with Windows, Quit
- Starts Streamlit silently in background
- Opens browser automatically
"""

import sys
import os
import subprocess
import threading
import socket
import webbrowser
import time
import platform
from pathlib import Path

# ── Platform-specific imports ────────────────────────────────────────────────
IS_WINDOWS = platform.system() == 'Windows'
if IS_WINDOWS:
    import winreg

# ── Ensure we're in the app directory ────────────────────────────────────────
APP_DIR = Path(__file__).parent.resolve()
os.chdir(APP_DIR)
sys.path.insert(0, str(APP_DIR))

PORT = 8501
URL  = f"http://localhost:{PORT}"

# ── Generate the tray icon as a simple PNG using PIL ─────────────────────────
def make_icon():
    """Load icon from udip_icon.png, or generate fallback."""
    from PIL import Image, ImageDraw
    
    # Try loading pre-built icon
    icon_file = APP_DIR / "udip_icon.png"
    if icon_file.exists():
        try:
            img = Image.open(str(icon_file)).convert("RGBA")
            return img.resize((64, 64), Image.LANCZOS)
        except Exception:
            pass
    
    # Fallback: generate icon
    size = 64
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([2, 2, size-2, size-2], radius=10, fill=(20, 80, 160))
    for bar in [(8,44,16,56),(20,32,28,56),(32,20,40,56),(44,36,52,56)]:
        draw.rectangle(bar, fill=(255,255,255))
    return img

# ── Check if server is up ─────────────────────────────────────────────────────
def is_server_running() -> bool:
    try:
        with socket.create_connection(("localhost", PORT), timeout=1):
            return True
    except (ConnectionRefusedError, OSError):
        return False

# ── Streamlit server process ──────────────────────────────────────────────────
_server_process = None

def start_server():
    global _server_process

    # Try pythonw.exe first (no window), fall back to python.exe
    python = APP_DIR / "venv" / "Scripts" / "pythonw.exe"
    if not python.exists():
        # Try system Python folder
        import shutil
        system_pythonw = shutil.which("pythonw")
        if system_pythonw:
            python = Path(system_pythonw)
        else:
            python = APP_DIR / "venv" / "Scripts" / "python.exe"
    if not python.exists():
        python = Path(sys.executable)

    streamlit = APP_DIR / "venv" / "Scripts" / "streamlit.exe"

    cmd = [
        str(streamlit), "run", str(APP_DIR / "app.py"),
        "--server.port", str(PORT),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--server.enableXsrfProtection", "false",
        "--logger.level", "error",
    ]

    _server_process = subprocess.Popen(
        cmd,
        cwd=str(APP_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )

def stop_server():
    global _server_process
    if _server_process:
        _server_process.terminate()
        try:
            _server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_process.kill()
        _server_process = None

def restart_server(icon, item):
    icon.notify("Restarting UDIP...", "UDIP v3")
    stop_server()
    time.sleep(2)
    start_server()
    wait_and_open()

# ── Browser opener ────────────────────────────────────────────────────────────
def wait_and_open():
    """Wait for server to be ready then open browser."""
    for _ in range(30):  # wait up to 15 seconds
        if is_server_running():
            webbrowser.open(URL)
            return
        time.sleep(0.5)

def open_browser(icon, item):
    if is_server_running():
        webbrowser.open(URL)
    else:
        icon.notify("Starting UDIP...", "UDIP v3")
        start_server()
        wait_and_open()

# ── Windows startup (run with Windows) ───────────────────────────────────────
if IS_WINDOWS:
    STARTUP_KEY  = r"Software\Microsoft\Windows\CurrentVersion\Run"
    STARTUP_NAME = "UDIP_v3"

    def is_startup_enabled() -> bool:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, STARTUP_KEY) as k:
                winreg.QueryValueEx(k, STARTUP_NAME)
                return True
        except FileNotFoundError:
            return False

    def set_startup(enable: bool):
        vbs_launcher = APP_DIR / "Launch_UDIP_Silent.vbs"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, STARTUP_KEY, 0,
                            winreg.KEY_SET_VALUE) as k:
            if enable:
                cmd = f'wscript.exe "{vbs_launcher}"'
                winreg.SetValueEx(k, STARTUP_NAME, 0, winreg.REG_SZ, cmd)
            else:
                try:
                    winreg.DeleteValue(k, STARTUP_NAME)
                except FileNotFoundError:
                    pass

    def toggle_startup(icon, item):
        current = is_startup_enabled()
        set_startup(not current)
        msg = "UDIP will now start with Windows" if not current else "UDIP will no longer start with Windows"
        icon.notify(msg, "UDIP v3")
        # Rebuild menu to reflect new state
        icon.menu = make_menu(icon)
else:
    def is_startup_enabled() -> bool:
        return False

    def toggle_startup(icon, item):
        pass

# ── Status check ─────────────────────────────────────────────────────────────
def get_status(icon, item):
    if is_server_running():
        icon.notify(f"UDIP is running at {URL}", "UDIP v3 — Online")
    else:
        icon.notify("UDIP server is not running. Click Open to start.", "UDIP v3 — Offline")

# ── Quit ──────────────────────────────────────────────────────────────────────
def quit_app(icon, item):
    icon.notify("UDIP is shutting down...", "UDIP v3")
    time.sleep(1)
    stop_server()
    icon.stop()

# ── Build menu ────────────────────────────────────────────────────────────────
def make_menu(icon=None):
    import pystray
    status_label  = f"● Running at {URL}" if is_server_running() else "○ Not running"

    menu_items = [
        pystray.MenuItem("📊 Open UDIP",          open_browser, default=True),
        pystray.MenuItem(status_label,             get_status),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("🔄 Restart server",      restart_server),
    ]

    if IS_WINDOWS:
        startup_label = "✓ Start with Windows" if is_startup_enabled() else "  Start with Windows"
        menu_items.append(pystray.MenuItem(startup_label, toggle_startup))

    menu_items.append(pystray.Menu.SEPARATOR)
    menu_items.append(pystray.MenuItem("✖ Quit UDIP", quit_app))

    return pystray.Menu(*menu_items)

# ── Main ──────────────────────────────────────────────────────────────────────
def force_tray_icon_visible():
    """Tell Windows to always show this tray icon, never hide it."""
    if not IS_WINDOWS:
        return
    try:
        import winreg, struct
        # Windows stores tray icon visibility in this key
        key_path = r"Software\Microsoft\Windows\CurrentVersion\Explorer\TrayNotify"
        
        # The easier approach: just set the notification area to show all icons
        settings_key = r"Software\Microsoft\Windows\CurrentVersion\Explorer\StuckRects3"
        # This sets taskbar to "Always show all icons" in notification area
        notify_path = r"Software\Microsoft\Windows\CurrentVersion\Explorer\TrayNotify"
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, notify_path, 0, winreg.KEY_SET_VALUE) as k:
                winreg.SetValueEx(k, "SystemTrayChevronInfoAutoOpen", 0, winreg.REG_DWORD, 1)
        except Exception:
            pass
    except Exception:
        pass  # Best effort — not critical


def kill_existing_instances():
    """Kill any existing UDIP tray processes before starting a new one."""
    import psutil, os
    current_pid = os.getpid()
    tray_script = str(APP_DIR / "tray.py")
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.pid == current_pid:
                continue
            cmdline = proc.info.get('cmdline') or []
            if any('tray.py' in str(c) for c in cmdline):
                proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def main():
    import pystray

    # Kill duplicate instances
    try:
        kill_existing_instances()
    except ImportError:
        pass  # psutil not installed - skip duplicate check

    force_tray_icon_visible()

    icon_img = make_icon()

    # Start server in background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Open browser after server starts
    browser_thread = threading.Thread(target=wait_and_open, daemon=True)
    browser_thread.start()

    # Create tray icon
    tray_icon = pystray.Icon(
        name="UDIP",
        icon=icon_img,
        title="UDIP v3 — Universal Data Integration Platform",
        menu=make_menu(),
    )

    # Show startup notification
    def notify_when_ready():
        time.sleep(3)
        if is_server_running():
            tray_icon.notify(
                "UDIP is ready! \n"
                "Tip: drag the UDIP icon from the ^ arrow to your taskbar to keep it visible.",
                "UDIP v3 — Running"
            )

    threading.Thread(target=notify_when_ready, daemon=True).start()

    # Run tray (blocks until quit)
    tray_icon.run()


if __name__ == "__main__":
    main()
