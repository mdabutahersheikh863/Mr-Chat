#!/usr/bin/env python3
"""Nightly summary for the Mr Support bot (feature 6).

Reads today's activity from data/history.json, data/users.json and
logs/bot.log, asks Groq (Pollinations fallback) to summarise the
day's topics in Bengali, and sends it to the admin via the Bot API.

Usage:
  python3 daily_summary.py            # send tonight's summary
  python3 daily_summary.py --dry-run  # print it, send nothing

Run with: env -u no_proxy -u NO_PROXY python3 daily_summary.py
(the VM's no_proxy breaks Telegram's httpx client otherwise).
"""
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
LOG_FILE = BASE / "logs" / "bot.log"
DHAKA = ZoneInfo("Asia/Dhaka")

OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct"


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


def load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return default


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
    ai_replies = 0
    backends = {}
    if not LOG_FILE.exists():
        return users, to_admin, ai_replies, backends
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
        elif body.startswith("ai reply to user"):
            ai_replies += 1
            if um:
                users.add(um.group(1))
        elif body.startswith("ai reply via "):
            name = body[len("ai reply via "):].strip()
            backends[name] = backends.get(name, 0) + 1
    return users, to_admin, ai_replies, backends


def todays_texts(day_start_ts: int):
    """User message texts from today (history keeps last 10/user)."""
    hist = load_json(DATA_DIR / "history.json", {})
    texts = []
    for _uid, items in hist.items():
        for it in items:
            if it.get("role") == "user" and it.get("ts", 0) >= day_start_ts:
                texts.append(it.get("text", "")[:200])
    return texts[:40]


def summarise_topics(texts):
    """Short Bengali topic summary via Groq, Pollinations fallback."""
    if not texts:
        return "আজ কোনো মেসেজ আসেনি।"
    joined = "\n".join(f"- {t}" for t in texts)
    prompt = ("Summarise these user messages from today in Bengali, "
              "in 3-5 short bullet lines: which topics/problems came "
              "up most, and anything urgent. Be concrete, no fluff.\n\n"
              + joined)
    # Groq first (same pattern as bot.py openrouter_reply).
    try:
        sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
        from dynamic_credentials import add_surrogate_to_request
        body = json.dumps({
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 400,
            "provider": {"order": ["groq"], "allow_fallbacks": False},
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        add_surrogate_to_request(req, "custom.openrouter",
                                 allowed_hosts=["openrouter.ai"])
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        ans = data["choices"][0]["message"]["content"].strip()
        if ans:
            return ans[:1200]
    except Exception as e:
        print(f"groq summary failed: {e}", file=sys.stderr)
    # Pollinations fallback.
    try:
        body = json.dumps({
            "messages": [
                {"role": "system",
                 "content": "Answer briefly in Bengali."},
                {"role": "user", "content": prompt},
            ],
            "model": "openai", "temperature": 0,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://text.pollinations.ai/", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            ans = resp.read().decode("utf-8", "ignore").strip()
        if ans:
            return ans[:1200]
    except Exception as e:
        print(f"pollinations summary failed: {e}", file=sys.stderr)
    # Last resort: raw topic lines.
    return "\n".join(f"• {t[:120]}" for t in texts[:8])


def build_summary() -> str:
    now_dhaka = datetime.now(DHAKA)
    start_utc, end_utc = day_bounds_utc(now_dhaka)
    users, to_admin, ai_replies, backends = count_from_log(start_utc,
                                                           end_utc)
    day_start_ts = int(start_utc.timestamp())
    texts = todays_texts(day_start_ts)
    topics = summarise_topics(texts)
    date_bn = now_dhaka.strftime("%d %B %Y")
    backend_line = ""
    if backends:
        parts = [f"{k}: {v}" for k, v in sorted(backends.items())]
        backend_line = f" ({', '.join(parts)})"
    return (
        f"📊 <b>আজকের সারাংশ — {date_bn}</b>\n\n"
        f"👥 মেসেজ পাঠিয়েছে: {len(users)} জন\n"
        f"📨 এডমিনের কাছে গেছে: {to_admin} টি\n"
        f"🤖 AI উত্তর দিয়েছে: {ai_replies} টি{backend_line}\n\n"
        f"📌 <b>মূল বিষয়গুলো:</b>\n{topics}"
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
