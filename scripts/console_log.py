"""Stream a CLI process to both the terminal and an append-only console log."""

from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys


def run_logged(command: list[str], log_path: Path, *, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log:
        heading = f"\n--- Run started {datetime.now(timezone.utc).isoformat()} ---\n"
        log.write(heading.encode())
        print(f"Console output: {log_path}", flush=True)
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, start_new_session=True,
        )
        assert process.stdout is not None
        try:
            while chunk := process.stdout.read1(65536):
                log.write(chunk)
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            code = process.wait()
        except KeyboardInterrupt:
            # Notebook interrupts reach this wrapper; forward them to the
            # collector and its workers rather than leaving GPU jobs running.
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            remainder = process.stdout.read()
            log.write(remainder)
            sys.stdout.buffer.write(remainder)
            sys.stdout.buffer.flush()
            code = 130
        finally:
            process.stdout.close()
        code = code if code >= 0 else 128 - code
        log.write(f"\n--- Run exited with code {code} ---\n".encode())
        return code
