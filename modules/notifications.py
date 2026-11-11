"""
UDIP v3 — modules/notifications.py
Windows toast notifications for long-running tasks.
"""

import threading
import time
import os
from pathlib import Path


def _send_toast(title: str, message: str, duration: int = 5):
    """Send a Windows toast notification. Runs in background thread."""
    try:
        # Method 1: Windows 10/11 native toast via PowerShell
        ps_script = f'''
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null

$template = @"
<toast>
  <visual>
    <binding template="ToastGeneric">
      <text>{title}</text>
      <text>{message}</text>
    </binding>
  </visual>
</toast>
"@

$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml($template)
$toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
$notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("UDIP v3")
$notifier.Show($toast)
'''
        import subprocess
        subprocess.run(
            ["powershell", "-WindowStyle", "Hidden", "-Command", ps_script],
            capture_output=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        )
        return
    except Exception:
        pass

    try:
        # Method 2: plyer (cross-platform)
        from plyer import notification
        notification.notify(
            title=title,
            message=message,
            app_name="UDIP v3",
            timeout=duration,
        )
        return
    except Exception:
        pass

    try:
        # Method 3: win10toast fallback
        from win10toast import ToastNotifier
        toaster = ToastNotifier()
        toaster.show_toast(title, message, duration=duration, threaded=True)
        return
    except Exception:
        pass


def notify(title: str, message: str, duration: int = 5):
    """
    Send a Windows notification in a background thread.
    Non-blocking — returns immediately.
    """
    if os.name != "nt":
        return  # Only on Windows
    t = threading.Thread(target=_send_toast, args=(title, message, duration), daemon=True)
    t.start()


def notify_task_complete(task_name: str, details: str = "", success: bool = True):
    """Notify that a task finished."""
    icon = "✅" if success else "❌"
    title = f"{icon} UDIP — {task_name} complete"
    msg   = details or ("Finished successfully" if success else "Finished with errors")
    notify(title, msg)


def notify_with_timer(task_name: str, threshold_seconds: float = 5.0):
    """
    Context manager — only sends notification if task takes longer than threshold.
    
    Usage:
        with notify_with_timer("Merge datasets", threshold_seconds=3):
            result = run_long_operation()
    """
    return _TimedNotification(task_name, threshold_seconds)


class _TimedNotification:
    def __init__(self, task_name: str, threshold: float):
        self.task_name = task_name
        self.threshold = threshold
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if elapsed >= self.threshold:
            success = exc_type is None
            duration_str = f"{elapsed:.1f}s"
            if elapsed >= 60:
                duration_str = f"{elapsed/60:.1f}min"
            notify_task_complete(
                self.task_name,
                details=f"Completed in {duration_str}",
                success=success,
            )
        return False  # Don't suppress exceptions