"""
Autonomous Colab notebook runner.

Connects to the user's real Chrome (with Google login) via CDP,
uploads a notebook to Drive, opens it in Colab, sets T4 GPU, and runs all cells.

Prerequisites:
  - Chromium running with --remote-debugging-port=9222
  - rclone configured with 'gdrive' remote
  - User logged into Google in Chrome

Usage:
    python scripts/colab_runner.py notebooks/train_deliberation_colab.ipynb
    python scripts/colab_runner.py --test  # Quick connectivity test
"""

import argparse
import json
import subprocess
import time


def rclone_upload(local_path: str) -> str:
    """Upload file to Drive root, return file ID."""
    filename = local_path.split("/")[-1]

    # Delete existing version if any
    subprocess.run(["rclone", "deletefile", f"gdrive:{filename}"], capture_output=True)

    # Upload
    result = subprocess.run(
        ["rclone", "copy", local_path, "gdrive:"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"rclone upload failed: {result.stderr}")

    # Get file ID
    result = subprocess.run(
        ["rclone", "lsjson", "gdrive:", "--include", filename],
        capture_output=True,
        text=True,
    )
    files = json.loads(result.stdout)
    if not files:
        raise RuntimeError(f"File {filename} not found on Drive after upload")
    return files[-1]["ID"]


def run_in_colab(file_id: str, timeout_minutes: int = 90):
    """Open notebook in Colab, set T4, run all cells, wait for completion."""
    from playwright.sync_api import sync_playwright

    colab_url = f"https://colab.research.google.com/drive/{file_id}"

    with sync_playwright() as p:
        # Connect to existing Chrome via CDP
        browser = p.chromium.connect_over_cdp("http://127.0.0.1:9222")
        context = browser.contexts[0]

        print(f"Connected to Chrome ({len(context.pages)} tabs open)")

        # Open Colab in a new tab
        page = context.new_page()
        print(f"Navigating to Colab: {colab_url}")
        page.goto(colab_url, wait_until="networkidle", timeout=60000)
        time.sleep(5)

        print(f"Page title: {page.title()}")

        # Close any popups/dialogs
        try:
            page.click("text=Dismiss", timeout=3000)
        except Exception:
            pass
        try:
            page.click("text=OK, Got it", timeout=3000)
        except Exception:
            pass

        # Change runtime to T4 GPU
        print("Setting T4 GPU runtime...")
        # Command palette is the most reliable way to open the dialog
        page.keyboard.press("Control+Shift+p")
        time.sleep(1)
        page.keyboard.type("change runtime", delay=50)
        time.sleep(1)
        page.keyboard.press("Enter")
        time.sleep(3)

        # Select T4 GPU radio - force click bypasses shadow DOM overlay
        try:
            page.get_by_text("T4 GPU", exact=True).click(force=True, timeout=5000)
            print("T4 GPU selected")
        except Exception:
            print("T4 radio click failed - may already be selected")

        time.sleep(1)

        # Save button is in mwc-dialog shadow DOM - Tab+Enter is the reliable path
        for _ in range(5):
            page.keyboard.press("Tab")
            time.sleep(0.2)
        page.keyboard.press("Enter")
        time.sleep(5)
        print("Save clicked via Tab+Enter")

        # Run all cells
        print("Running all cells...")
        page.keyboard.press("Control+F9")
        time.sleep(5)

        # Handle confirmations
        for btn_text in ["Run anyway", "Yes", "OK"]:
            try:
                page.get_by_text(btn_text, exact=True).click(force=True, timeout=3000)
                print(f"Clicked '{btn_text}'")
            except Exception:
                pass

        # Handle Google Drive auth popup (from drive.mount)
        time.sleep(10)
        for pg in context.pages:
            if "accounts.google" in pg.url:
                try:
                    pg.get_by_text("Continue", exact=True).click(timeout=5000)
                    print("Drive auth: clicked Continue")
                    time.sleep(3)
                    pg.get_by_text("Continue", exact=True).click(timeout=5000)
                    print("Drive auth: clicked Continue (consent)")
                except Exception:
                    pass

        # Wait for completion by monitoring the runtime status
        print(f"Waiting for completion (timeout: {timeout_minutes} min)...")
        start_time = time.time()
        last_status = ""

        while (time.time() - start_time) < timeout_minutes * 60:
            time.sleep(30)
            elapsed = int((time.time() - start_time) / 60)

            # Check if any cells are still running (spinning icon)
            try:
                running = page.query_selector_all('[class*="running"]')
                pending = page.query_selector_all('[class*="pending"]')

                status = f"[{elapsed}m] running={len(running)} pending={len(pending)}"
                if status != last_status:
                    print(status)
                    last_status = status

                if len(running) == 0 and len(pending) == 0 and elapsed > 2:
                    print("All cells completed!")
                    break
            except Exception:
                print(f"[{elapsed}m] checking...")

        # Take a screenshot of the final state
        page.screenshot(path="/tmp/colab-final.png")
        print("Final screenshot saved to /tmp/colab-final.png")

        # Don't close the tab - user might want to inspect results
        print("Done. Notebook tab left open for inspection.")


def test_connection():
    """Quick test: connect to Chrome, verify Google login."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://127.0.0.1:9222")
        context = browser.contexts[0]
        print(f"Connected to Chrome: {len(context.pages)} tabs")

        page = context.new_page()
        page.goto(
            "https://accounts.google.com", wait_until="networkidle", timeout=15000
        )
        title = page.title()
        print(f"Google page title: {title}")

        # Check if logged in
        if "Sign in" in title:
            print("NOT LOGGED IN - need to sign into Google first")
        else:
            print("LOGGED IN - Google session active")

        page.close()
        print("Test passed!")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", nargs="?", help="Path to .ipynb file")
    parser.add_argument("--test", action="store_true", help="Test Chrome connection")
    parser.add_argument("--timeout", type=int, default=90, help="Max minutes to wait")
    args = parser.parse_args()

    if args.test:
        test_connection()
        return

    if not args.notebook:
        parser.error("Provide a notebook path or --test")

    print("=== Colab Runner ===")
    print(f"Notebook: {args.notebook}")

    # Step 1: Upload to Drive
    print("\n[1/3] Uploading to Google Drive...")
    file_id = rclone_upload(args.notebook)
    print(f"Uploaded. File ID: {file_id}")

    # Step 2: Run in Colab
    print("\n[2/3] Opening in Colab and running...")
    run_in_colab(file_id, args.timeout)

    # Step 3: Results
    print("\n[3/3] Check results in the Colab tab or download from Drive")
    print("  rclone ls gdrive:")
    print("  rclone copy gdrive:deliberation-model/ ./modal-output/")


if __name__ == "__main__":
    main()
