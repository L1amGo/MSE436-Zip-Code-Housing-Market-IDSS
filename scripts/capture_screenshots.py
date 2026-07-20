"""Capture dashboard screenshots for the report and slides (task D5).

    pip install playwright && playwright install chromium
    python scripts/capture_screenshots.py

Launches the dashboard headless, drives the real controls, and writes PNGs to
reports/figures/. The before/after pair is the important one: it is the evidence
that the controls change model output rather than filtering a static table, so
both frames are captured from the same session with one control moved between
them.

Playwright is a capture-time dependency only — the dashboard itself does not
need it.
"""

from __future__ import annotations

import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.io_utils import load_config  # noqa: E402

PORT = 8701
VIEWPORT = {"width": 1680, "height": 1050}
SCALE = 2  # retina: legible when dropped into a slide
DEMO_METRO = "New Haven, CT"
BEFORE_BPS = "No change"
AFTER_BPS = "+100 bps"


def wait_for_port(port: int, timeout: int = 180) -> bool:
    for _ in range(timeout):
        try:
            socket.create_connection(("localhost", port), timeout=1).close()
            return True
        except OSError:
            time.sleep(1)
    return False


def scroll(page, pixels: int) -> None:
    """Scroll the app. The cursor is parked on the far right because Plotly
    swallows wheel events that land on a chart."""
    page.mouse.move(VIEWPORT["width"] - 12, 300)
    moved = 0
    while moved < pixels:
        page.mouse.wheel(0, 600)
        moved += 600
        page.wait_for_timeout(250)
    page.wait_for_timeout(2500)


def select_metro(page, metro: str) -> None:
    page.locator("[data-testid='stMultiSelect']").first.click()
    page.wait_for_timeout(600)
    page.keyboard.type(metro, delay=25)
    page.wait_for_timeout(1200)
    page.keyboard.press("Enter")
    page.wait_for_timeout(2500)
    page.keyboard.press("Escape")
    page.wait_for_timeout(1500)


def choose_scenario(page, label: str) -> None:
    """Click a radio in the mortgage-rate scenario group by its visible label."""
    page.get_by_text(label, exact=True).first.click()
    page.wait_for_timeout(4000)


# Sliders in sidebar order: min ROI, max downside, risk tolerance.
RISK_SLIDER_INDEX = 2


def set_risk_tolerance(page, end: str) -> None:
    """Drive the risk-tolerance select_slider to its first or last option.

    Home/End on a focused slider is stabler than computing a pixel offset for
    the thumb, which moves with the sidebar width.
    """
    slider = page.locator("[data-testid='stSlider']").nth(RISK_SLIDER_INDEX)
    slider.click()
    page.wait_for_timeout(400)
    page.keyboard.press("Home" if end == "first" else "End")
    page.wait_for_timeout(4000)


def capture(out_dir: Path) -> list[Path]:
    from playwright.sync_api import sync_playwright

    written: list[Path] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=SCALE)
        page.goto(f"http://localhost:{PORT}", wait_until="networkidle", timeout=180_000)
        page.wait_for_selector("[data-testid='stDataFrame']", timeout=240_000)
        page.wait_for_timeout(5000)

        # 1. Full dashboard: header metrics, decision summary, ranked table.
        full = out_dir / "dashboard_full.png"
        page.screenshot(path=str(full))
        written.append(full)

        # 2. The control panel on its own.
        sidebar = out_dir / "dashboard_sidebar.png"
        page.locator("[data-testid='stSidebar']").first.screenshot(path=str(sidebar))
        written.append(sidebar)

        # 3. Before/after: one control moved, everything else identical.
        before = out_dir / "scenario_before.png"
        page.screenshot(path=str(before))
        written.append(before)

        choose_scenario(page, AFTER_BPS)
        after = out_dir / "scenario_after.png"
        page.screenshot(path=str(after))
        written.append(after)

        choose_scenario(page, BEFORE_BPS)

        # 4. Risk-tolerance pair. The rate scenario moves a nationwide macro
        #    input, so it shifts predictions and eligibility but barely re-orders
        #    (Spearman 0.997). Risk tolerance is the control that genuinely
        #    re-ranks (0.79), so the pair that evidences re-ranking is captured
        #    separately rather than overclaiming the scenario pair.
        set_risk_tolerance(page, "first")  # Low
        risk_low = out_dir / "risk_low.png"
        page.screenshot(path=str(risk_low))
        written.append(risk_low)

        set_risk_tolerance(page, "last")  # High
        risk_high = out_dir / "risk_high.png"
        page.screenshot(path=str(risk_high))
        written.append(risk_high)

        # 5. Drill-down, which needs a metro selected so the page is short
        #    enough to reach and the map has something to draw.
        select_metro(page, DEMO_METRO)
        try:
            page.wait_for_selector(".js-plotly-plot", timeout=180_000)
        except Exception:
            pass
        page.wait_for_timeout(8000)
        scroll(page, 8400)
        drill = out_dir / "drilldown.png"
        page.screenshot(path=str(drill))
        written.append(drill)

        for el in page.query_selector_all("[data-testid='stException']"):
            print("PAGE EXCEPTION:", el.inner_text()[:500], file=sys.stderr)

        browser.close()
    return written


def main() -> int:
    config = load_config()
    out_dir = REPO_ROOT / config["paths"]["reports"] / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import playwright  # noqa: F401
    except ImportError:
        print(
            "playwright is not installed. Run:\n"
            "    pip install playwright && playwright install chromium",
            file=sys.stderr,
        )
        return 1

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "app/main.py",
            "--server.headless", "true", "--server.port", str(PORT),
            "--server.fileWatcherType", "none",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    try:
        if not wait_for_port(PORT):
            print(f"dashboard did not start on port {PORT}", file=sys.stderr)
            return 1
        written = capture(out_dir)
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

    for path in written:
        size_kb = path.stat().st_size / 1024
        print(f"wrote {path.relative_to(REPO_ROOT)} ({size_kb:,.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
