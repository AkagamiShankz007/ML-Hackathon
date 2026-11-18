"""Capture real screenshots of the running RouteLens UI (needs the API on :8000 and the UI on :5173; uses the installed Chrome).
Usage: python scripts/capture_ui_screenshots.py [trip_id ...]   -> artifacts/ui_screenshots/*.png"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "ui_screenshots"; OUT.mkdir(parents=True, exist_ok=True)
trips = sys.argv[1:] or ["00000009", "00000056"]
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True); pg = b.new_page(viewport={"width": 1500, "height": 900})
    pg.goto("http://localhost:5173"); pg.wait_for_selector("#trips li"); pg.wait_for_timeout(1500)
    pg.screenshot(path=OUT / "01_trip_list.png")
    for t in trips:
        pg.locator("#trips li", has_text=t).click(); pg.wait_for_timeout(1500)
        pg.check("#lNet"); pg.wait_for_timeout(2500)
        pg.click("#run"); pg.wait_for_selector("#results:not([hidden])", timeout=120000); pg.wait_for_timeout(4000)
        pg.screenshot(path=OUT / f"02_trip_{t}_hmm_compare.png")
        pg.uncheck("#lNet")
    b.close()
print("saved:", sorted(x.name for x in OUT.iterdir()))
