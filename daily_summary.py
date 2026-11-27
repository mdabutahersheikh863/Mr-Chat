#!/usr/bin/env python3
"""Nightly summary for the Mr Support bot (feature 6).

Counts-only (no AI since 2026-11-27, when the AI was removed from
the bot on Abu's order): reads today's activity from logs/bot.log
and sends the counts to the admin via the Bot API.

Usage:
  python3 daily_summary.py            # send tonight's summary
  python3 daily_summary.py --dry-run  # print it, send nothing

Run with: env -u no_proxy -u NO_PROXY python3 daily_summary.py
(the VM's no_proxy breaks Telegram's httpx client otherwise).
"""
import json
import re
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parent
LOG_FILE = BASE / "logs" / "bot.log"
DHAKA = ZoneInfo("Asia/Dhaka")


def load_env(path: Path) -> dict:
    env = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


ENV = load_env(BASE / ".env")
TOKEN = ENV.get("BOT_TOKEN", "")
ADMIN_ID = int(ENV.get("ADMIN_ID", "0") or 0)

# Token source mirrors bot.py: when the marker data/use_vault_token
# exists, the live bot runs on the NEW bot's vault token
# (@Mr_Chat1bot) and the old .env token is dead (401). The summary
# must send with the same token the bot polls with. The surrogate
# goes RAW in the request path (urllib does not re-encode the colon).
VAULT_TOKEN_MARKER = BASE / "data" / "use_vault_token"
if VAULT_TOKEN_MARKER.exists():
    try:
        sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
        from dynamic_credentials import dynamic_credential_entry
        TOKEN = dynamic_credential_entry(
            "custom.telegram-bot")["surrogate"]
    except Exception as _exc:
        raise SystemExit(f"vault token requested but unavailable: {_exc}")


def day_bounds_utc(now_dhaka: datetime):
    """Start/end of the Dhaka calendar day, as UTC datetimes."""
    start_dh = now_dhaka.replace(hour=0, minute=0, second=0, microsecond=0)
    end_dh = start_dh + timedelta(days=1)
    return start_dh.astimezone(timezone.utc), end_dh.astimezone(timezone.utc)


LOG_LINE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ "
    r"\w+ contact-bot: (.*)$")
USER_RE = re.compile(r"user (\d+)")


def count_from_log(start_utc, end_utc):
    """Count today's activity from bot.log (timestamps are UTC)."""
    users = set()
    to_admin = 0
    if not LOG_FILE.exists():
        return users, to_admin
    for line in LOG_FILE.read_text(encoding="utf-8",
                                   errors="ignore").splitlines():
        m = LOG_LINE.match(line)
        if not m:
            continue
        try:
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        ts = ts.replace(tzinfo=timezone.utc)
        if not (start_utc <= ts < end_utc):
            continue
        body = m.group(2)
        um = USER_RE.search(body)
        if body.startswith("sent message from user"):
            to_admin += 1
            if um:
                users.add(um.group(1))
    return users, to_admin


def build_summary() -> str:
    now_dhaka = datetime.now(DHAKA)
    start_utc, end_utc = day_bounds_utc(now_dhaka)
    users, to_admin = count_from_log(start_utc, end_utc)
    date_bn = now_dhaka.strftime("%d %B %Y")
    return (
        f"📊 <b>আজকের সারাংশ — {date_bn}</b>\n\n"
        f"👥 মেসেজ পাঠিয়েছে: {len(users)} জন\n"
        f"📨 এডমিনের কাছে গেছে: {to_admin} টি"
    )


def send_to_admin(text: str) -> None:
    body = json.dumps({"chat_id": ADMIN_ID, "text": text,
                       "parse_mode": "HTML"}).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if not data.get("ok"):
        raise RuntimeError(f"telegram send failed: {data}")


def main() -> None:
    dry = "--dry-run" in sys.argv
    if not TOKEN or not ADMIN_ID:
        print("BOT_TOKEN or ADMIN_ID missing in .env", file=sys.stderr)
        raise SystemExit(1)
    text = build_summary()
    if dry:
        print(text)
        return
    send_to_admin(text)
    print("summary sent to admin")


if __name__ == "__main__":
    main()
