# tools/software.py

import subprocess, requests

class SoftwareTool:
    def run(self, *args, **kwargs):
        raise NotImplementedError

class ShellTool(SoftwareTool):
    def run(self, cmd: str, timeout: int = 5):
        proc = subprocess.run(cmd, shell=True, capture_output=True, timeout=timeout, text=True)
        return {"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}

class DesktopAutomationTool(SoftwareTool):
    def __init__(self):
        import pyautogui
        self._py = pyautogui

    def run(self, action: str, **params):
        # e.g. action="click", params={"x":100,"y":200}
        func = getattr(self._py, action)
        result = func(**params)
        return {"result": result}

class RestAPITool(SoftwareTool):
    def run(self, method: str, url: str, **kwargs):
        resp = requests.request(method, url, **kwargs)
        return {"status": resp.status_code, "body": resp.text}
