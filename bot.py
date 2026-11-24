#!/usr/bin/env python3
"""Admin contact bot v6 (inbox only, no mini app, no inline buttons).

- ReplyKeyboardMarkup buttons only, NO emoji on buttons.
- Button labels use Mathematical Sans-Serif Bold Unicode
  (𝐀-𝐙, 𝐚-𝐳, 𝟎-𝟗). Bengali text has no such bold form, so
  message bodies stay in normal Bengali.
- Button colours via Bot API 9.4 `style` (primary = blue,
  success = green, danger = red). Needs a Telegram client
  released after 9 Feb 2026; older clients show plain buttons.
- Connect-once session: tap "𝐓𝐚𝐥𝐤 𝐭𝐨 𝐀𝐝𝐦𝐢𝐧", then EVERY
  message goes to the admin until the user sends Stop / Cancel
  (button or typed word) -- then AI replies resume.
- User messages are NEVER forwarded. The admin gets one plain
  message in this format:
      নাম: first_name
      ইউজারনেম: @username
      চ্যাট আইডি: user_id
      মেসেজ👇
      <message>
  The admin replies directly to that message and it is delivered
  to the user (admin-message-id -> user-id mapping, with a
  chat-ID regex fallback). /reply user_id text stays as backup.
- Auto AI reply chain: OpenRouter (Groq-served Llama 3.3 70B) when
  the custom.openrouter credential is connected, else Gemini when
  GEMINI_API_KEY is set in .env, else the free Pollinations text API
  (no key, model=openai, temperature=0, short English system prompt
  — 3/3 factual questions correct in test), else a built-in Bengali
  smart auto-reply.
- Admin broadcast: /broadcast text, or the Broadcast button
  followed by the message text.
- Owner identity: the bot knows its owner (intro + facts stored in
  data/bot_info.json). Admin commands: /setintro, /addinfo,
  /listinfo, /delinfo; anyone can see it with /owner or the Owner
  button. Owner questions ("এডমিন কে?") are answered from the
  intro instead of connecting to the admin.
- Conversation memory: the last 10 messages (user + bot) per user
  are kept in data/history.json and passed to the AI, so the next
  answer follows from what was already said.
- Smart support: the admin's reply texts are kept in
  data/admin_style.json (last 20) and shown to the AI as style
  examples, so answers sound like the admin's own replies.
- Voice notes are transcribed (OpenRouter gpt-audio-mini) and
  answered like text; photos are described by a vision model
  (Qwen2.5-VL 72B first — Llama 4 Scout is listed but returns 404
  on use — never a Google model). NOTE: OpenRouter audio calls
  need at least $0.50 balance for audio (HTTP 402 otherwise), so
  voice transcription stays blocked until the account is topped up;
  the bot answers "ভয়েস বুঝতে পারিনি" meanwhile.
- Every AI answer carries inline 👍/👎 buttons; ratings go to
  data/ratings.json and a 👎 is forwarded to the admin. The AI
  NEVER sends a voice message — AI replies are always text, even
  when the user sent a voice message (it is transcribed first).
- Urgent words (জরুরি, টাকা ফেরত, সমস্যা, ...) send the admin an
  immediate separate alert. Spam (>10 msgs/60s or the same message
  3x) auto-blocks the user (data/blocked.json); /unblock lifts it.
- Nightly summary at 23:55 via daily_summary.py + the
  mr-support-nightly-summary cron.
- Language button (🌐 Language) shows inline buttons to pick
  Bengali/English per user; the WHOLE bot (menus, messages, AI
  answers) follows it. AI answers are text only, never voice.
- /poll <question> sends a native Yes/No poll to all users;
  answers go to data/poll.json, /pollresults shows the tally.
- Admin offline 00:00-07:00 Dhaka: Talk to Admin says the admin
  is offline and to talk in the morning; AI stays on always.
- Each admin message carries an inline Translate button
  (Bengali<->English, Groq first, Pollinations fallback); tapping
  toggles the translation, the original stays. The admin has a
  "Text to Voice" toggle button (data/admin_voice.json): when ON,
  his typed replies reach the user as a voice message (local tts,
  text fallback if TTS fails); when OFF (default), replies go as
  plain text. Users have no text-to-voice button.
"""
import asyncio
import base64
import fcntl
import html
import io
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import httpx
from telegram.error import (
    Conflict,
    Forbidden,
    InvalidToken,
    NetworkError,
    RetryAfter,
)

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    PollAnswerHandler,
    filters,
)
from telegram.request import HTTPXRequest

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "data" / "users.json"
INFO_FILE = BASE / "data" / "bot_info.json"
HISTORY_FILE = BASE / "data" / "history.json"
HISTORY_LIMIT = 10
LOG_FILE = BASE / "logs" / "bot.log"


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
GEMINI_KEY = ENV.get("GEMINI_API_KEY", "")

# OpenRouter (custom.openrouter) serves the Groq model. The credential
# lives in the registered connection, never in a file: each call fetches
# an hsurr: surrogate from authd and lets it be swapped for the real
# key on the approved request to openrouter.ai.
# Model ID note (verified 2026-11-23): OpenRouter no longer lists a
# "groq/llama-3.3-70b-versatile" ID — the API answers 400 "not a valid
# model ID" and the /models list has zero groq/ IDs. Groq serves the
# same Llama 3.3 70B weights under the meta-llama ID; provider order
# ["groq"] pins the request to Groq (response field provider=Groq).
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct"
# Single-flight lock for OpenRouter calls (chat, translate, audio,
# vision): the free tier answers HTTP 402 "in-flight budget
# exhausted" when several requests run at once, so at most one
# OpenRouter request is in flight at a time.
_OPENROUTER_LOCK = threading.Lock()
try:
    sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
    from dynamic_credentials import add_surrogate_to_request
    OPENROUTER_AVAILABLE = True
except Exception:
    OPENROUTER_AVAILABLE = False

# New bot token (custom.telegram-bot): the token Abu gave via the secure
# page lives in the registered connection, never in a file. When the
# marker file data/use_vault_token exists, the bot loads an hsurr:
# surrogate from authd at startup and uses it as BOT_TOKEN; the egress
# proxy swaps it for the real token on requests to api.telegram.org.
# The surrogate must reach Telegram RAW in the request path — URL-quoting
# it breaks the swap (verified 2026-11-24: quoted -> Telegram 404,
# raw -> Telegram answers). PTB passes the token raw in the path, so
# polling works. Without the marker, .env BOT_TOKEN is used (old bot).
# If the marker exists but the vault token cannot be loaded, exit loudly
# instead of silently running the old bot.
VAULT_TOKEN_MARKER = BASE / "data" / "use_vault_token"
TOKEN_SOURCE = ".env"
if VAULT_TOKEN_MARKER.exists():
    try:
        sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
        from dynamic_credentials import dynamic_credential_entry
        TOKEN = dynamic_credential_entry(
            "custom.telegram-bot")["surrogate"]
        TOKEN_SOURCE = "vault"
    except Exception as _exc:
        raise SystemExit(f"vault token requested but unavailable: {_exc}")

DATA_FILE.parent.mkdir(exist_ok=True)
LOG_FILE.parent.mkdir(exist_ok=True)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"),
              logging.StreamHandler()],
)
# httpx logs full request URLs (which contain the bot token) at INFO;
# keep it at WARNING so the token never lands in the log file.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("contact-bot")


def load_data() -> dict:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"users": {}, "forwarded": 0, "threads": {}}


def save_data(data: dict) -> None:
    DATA_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                         encoding="utf-8")


DATA = load_data()
DATA.setdefault("threads", {})

# Per-user runtime state (in-memory; a restart clears it).
CONNECTED: set = set()          # user_ids in a live admin session
ADMIN_FLAGS = {"broadcast_pending": False}

ADMIN_WORDS = ["admin", "এডমিন", "অ্যাডমিন", "মানুষ", "অপারেটর",
               "সাপোর্ট", "কর্তৃপক্ষ", "মালিক"]
STOP_WORDS = {"stop", "cancel", "স্টপ", "বাতিল"}

# Feature 8: urgent keywords — a matching user message gets the admin
# an immediate separate alert on top of the normal admin message.
URGENT_WORDS = ["জরুরি", "জরুরী", "টাকা ফেরত", "সমস্যা", "বিপদ",
                "হেল্প", "urgent", "emergency", "refund", "ফেরত",
                "অভিযোগ", "complaint", "টাকা কাটছে", "টাকা",
                "প্রতারণা", "fraud", "scam", "help"]


def is_urgent(text: str) -> bool:
    t = (text or "").lower()
    return any(w.lower() in t for w in URGENT_WORDS)

OWNER_WORDS = ["এডমিন কে", "অ্যাডমিন কে", "মালিক কে", "বটের মালিক",
               "কার বট", "বটের ওনার", "ওনার কে", "owner"]


# ---------------- Owner identity (bot_info.json) ----------------
DEFAULT_INTRO = "আমি আবু তাহের — এই বটের মালিক ও এডমিন।"


def load_info() -> dict:
    if INFO_FILE.exists():
        try:
            info = json.loads(INFO_FILE.read_text(encoding="utf-8"))
            info.setdefault("intro", DEFAULT_INTRO)
            info.setdefault("facts", [])
            return info
        except Exception:
            pass
    return {"intro": DEFAULT_INTRO, "facts": []}


def save_info(info: dict) -> None:
    INFO_FILE.write_text(json.dumps(info, ensure_ascii=False, indent=2),
                         encoding="utf-8")


INFO = load_info()


def owner_plain() -> str:
    lines = [INFO.get("intro", DEFAULT_INTRO)]
    facts = INFO.get("facts", [])
    if facts:
        lines.append("জেনে রাখুন:")
        lines.extend(f"- {f}" for f in facts)
    return "\n".join(lines)


def owner_text(lang: str = "bn") -> str:
    """Owner identity card; the chrome translates, the admin's own
    intro/facts stay exactly as written."""
    if lang == "en":
        body = ("👑 <b>Bot Owner / Admin</b>\n\n"
                + html.escape(INFO.get("intro", DEFAULT_INTRO)))
        facts = INFO.get("facts", [])
        if facts:
            body += "\n\n📌 <b>Good to know:</b>\n" + "\n".join(
                f"• {html.escape(f)}" for f in facts)
        body += "\n\n💬 To talk, tap " + BTN_CONTACT + "."
        return body
    body = ("👑 <b>বটের মালিক / এডমিন</b>\n\n"
            + html.escape(INFO.get("intro", DEFAULT_INTRO)))
    facts = INFO.get("facts", [])
    if facts:
        body += "\n\n📌 <b>জেনে রাখুন:</b>\n" + "\n".join(
            f"• {html.escape(f)}" for f in facts)
    body += "\n\n💬 কথা বলতে চাইলে " + BTN_CONTACT + " চাপ দিন।"
    return body


def owner_context() -> str:
    """Short owner context appended to the AI system prompt."""
    ctx = ("Owner info (use it when asked about the owner/admin): "
           + INFO.get("intro", DEFAULT_INTRO))
    facts = INFO.get("facts", [])
    if facts:
        ctx += " Facts: " + "; ".join(facts)
    return ctx


# ---------------- Conversation memory (history.json) ----------------
# Last 10 messages (user + bot) per user, kept on disk so a restart
# does not forget them. Passed to the AI so the next answer follows
# from what was already said.

def load_history() -> dict:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_history(hist: dict) -> None:
    HISTORY_FILE.write_text(json.dumps(hist, ensure_ascii=False, indent=2),
                            encoding="utf-8")


HISTORY = load_history()


def get_history(user_id: int) -> list:
    return HISTORY.get(str(user_id), [])


def add_history(user_id: int, role: str, text: str) -> None:
    items = HISTORY.setdefault(str(user_id), [])
    items.append({"role": role, "text": text[:500],
                  "ts": int(time.time())})
    del items[:-HISTORY_LIMIT]
    save_history(HISTORY)


# ---------------- Admin style learning (feature 1) ----------------
# The admin's own reply texts (sent via reply-to or /reply) are kept
# here (last 20) and a few are shown to the AI as style examples, so
# the bot's answers gradually sound like the admin's own replies.
STYLE_FILE = BASE / "data" / "admin_style.json"
STYLE_LIMIT = 20


def load_style() -> list:
    if STYLE_FILE.exists():
        try:
            items = json.loads(STYLE_FILE.read_text(encoding="utf-8"))
            return items if isinstance(items, list) else []
        except Exception:
            pass
    return []


def save_style(items: list) -> None:
    STYLE_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2),
                          encoding="utf-8")


def record_admin_reply(text: str) -> None:
    text = (text or "").strip()
    if not text:
        return
    items = load_style()
    items.append({"text": text[:300], "ts": int(time.time())})
    del items[:-STYLE_LIMIT]
    try:
        save_style(items)
    except Exception as e:
        log.warning("saving admin style failed: %s", e)


def style_context() -> str:
    items = load_style()
    if not items:
        return ""
    examples = " | ".join(f'"{it["text"][:120]}"' for it in items[-3:])
    return (" The admin usually replies in this style: " + examples +
            " Match this tone and brevity when you can.")


# ---------------- Language choice (feature 14) ----------------
# Per-user language: "bn" (default) or "en". The WHOLE bot
# (menus, messages, AI answers) follows it; admin-facing strings
# stay Bengali.
LANG_FILE = BASE / "data" / "lang.json"


def load_lang() -> dict:
    if LANG_FILE.exists():
        try:
            items = json.loads(LANG_FILE.read_text(encoding="utf-8"))
            return items if isinstance(items, dict) else {}
        except Exception:
            pass
    return {}


def save_lang(langs: dict) -> None:
    LANG_FILE.write_text(json.dumps(langs, ensure_ascii=False, indent=2),
                         encoding="utf-8")


LANGS = load_lang()


def get_lang(user_id: int) -> str:
    return LANGS.get(str(user_id), "bn")


def set_lang(user_id: int, lang: str) -> None:
    LANGS[str(user_id)] = lang
    save_lang(LANGS)


# ---------------- Admin Text-to-Voice toggle ----------------
# Admin-only (Abu, 2026-11-24): when ON, the admin's typed replies
# reach the user as a voice message (local TTS, text fallback);
# when OFF (default), replies go as plain text. Persisted in
# data/admin_voice.json so the choice survives restarts. Users get
# no text-to-voice, and the AI never sends voice replies.
ADMIN_VOICE_FILE = BASE / "data" / "admin_voice.json"


def load_admin_voice() -> dict:
    if ADMIN_VOICE_FILE.exists():
        try:
            items = json.loads(ADMIN_VOICE_FILE.read_text(encoding="utf-8"))
            return items if isinstance(items, dict) else {}
        except Exception:
            pass
    return {}


def save_admin_voice(state: dict) -> None:
    ADMIN_VOICE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


ADMIN_VOICE = load_admin_voice()


def admin_voice_enabled() -> bool:
    return bool(ADMIN_VOICE.get("enabled", False))


def set_admin_voice(enabled: bool) -> None:
    ADMIN_VOICE["enabled"] = bool(enabled)
    save_admin_voice(ADMIN_VOICE)


# ---------------- Admin offline hours (feature 17) ----------------
# AI stays ON always. Only the Talk to Admin path is gated: between
# 00:00 and 07:00 Dhaka time the bot says the admin is offline
# instead of connecting.
DHAKA = ZoneInfo("Asia/Dhaka")


def admin_offline_now(now: datetime | None = None) -> bool:
    h = (now or datetime.now(DHAKA)).hour
    return 0 <= h < 7


OFFLINE_MSG = ("অ্যাডমিন এখন অফলাইনে আছেন, সকালে কথা বলতে পারবেন। 🙂\n"
               "আপনার প্রশ্ন থাকলে লিখুন — AI উত্তর দেবে।")


# ---------------- Survey / poll (feature 16) ----------------
# One active poll at a time. /poll sends a native Telegram poll to
# every user; answers land in data/poll.json; /pollresults shows
# the tally to the admin.
POLL_FILE = BASE / "data" / "poll.json"
POLL_OPTIONS = ["হ্যাঁ", "না"]


def load_poll() -> dict:
    if POLL_FILE.exists():
        try:
            items = json.loads(POLL_FILE.read_text(encoding="utf-8"))
            return items if isinstance(items, dict) else {}
        except Exception:
            pass
    return {}


def save_poll(poll: dict) -> None:
    POLL_FILE.write_text(json.dumps(poll, ensure_ascii=False, indent=2),
                         encoding="utf-8")


# ---------------- Spam auto-block (feature 11) ----------------
# >10 messages in 60s, or the same message 3x in 60s, gets the user
# auto-blocked (data/blocked.json). Blocked users are silently
# ignored afterwards; the admin unblocks with /unblock <user_id>.
BLOCKED_FILE = BASE / "data" / "blocked.json"
SPAM_WINDOW = 60.0
SPAM_MAX_MSGS = 10
SPAM_REPEAT = 3


def load_blocked() -> dict:
    if BLOCKED_FILE.exists():
        try:
            items = json.loads(BLOCKED_FILE.read_text(encoding="utf-8"))
            return items if isinstance(items, dict) else {}
        except Exception:
            pass
    return {}


def save_blocked(blocked: dict) -> None:
    BLOCKED_FILE.write_text(json.dumps(blocked, ensure_ascii=False,
                                       indent=2), encoding="utf-8")


BLOCKED = load_blocked()
_SPAM_TIMES: dict = {}   # user_id -> [monotonic timestamps in window]
_SPAM_TEXTS: dict = {}   # user_id -> {norm text: [count, last monotonic]}


def is_blocked(user_id: int) -> bool:
    return str(user_id) in BLOCKED


def check_spam(user_id: int, text: str) -> str | None:
    """Track one incoming message. Returns a block reason, or None."""
    now = time.monotonic()
    times = [t for t in _SPAM_TIMES.get(user_id, [])
             if now - t < SPAM_WINDOW]
    times.append(now)
    _SPAM_TIMES[user_id] = times
    if len(times) > SPAM_MAX_MSGS:
        return (f"{SPAM_MAX_MSGS}+ messages in "
                f"{int(SPAM_WINDOW)} seconds")
    norm = (text or "").strip().lower()
    if norm:
        per = _SPAM_TEXTS.setdefault(user_id, {})
        ent = per.get(norm)
        if ent is None or now - ent[1] > SPAM_WINDOW:
            per[norm] = [1, now]
        else:
            ent[0] += 1
            ent[1] = now
            if ent[0] >= SPAM_REPEAT:
                return (f"same message {SPAM_REPEAT} times in "
                        f"{int(SPAM_WINDOW)} seconds")
    return None


def block_user(user_id: int, reason: str) -> None:
    BLOCKED[str(user_id)] = {"reason": reason,
                             "at": int(time.time())}
    save_blocked(BLOCKED)


def plain_text(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s)


# ---------------- Mathematical Sans-Serif Bold ----------------
# Maps plain A-Z / a-z / 0-9 to 𝐀-𝐙 / 𝐚-𝐳 / 𝟎-𝟗. Every other
# character (incl. all Bengali) passes through unchanged.
_BOLD_UPPER = "𝐀𝐁𝐂𝐃𝐄𝐅𝐆𝐇𝐈𝐉𝐊𝐋𝐌𝐍𝐎𝐏𝐐𝐑𝐒𝐓𝐔𝐕𝐖𝐗𝐘𝐙"
_BOLD_LOWER = "𝐚𝐛𝐜𝐝𝐞𝐟𝐠𝐡𝐢𝐣𝐤𝐥𝐦𝐧𝐨𝐩𝐪𝐫𝐬𝐭𝐮𝐯𝐰𝐱𝐲𝐳"
_BOLD_DIGITS = "𝟎𝟏𝟐𝟑𝟒𝟓𝟔𝟕𝟖𝟗"
_PLAIN_ALNUM = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "abcdefghijklmnopqrstuvwxyz0123456789")
_BOLD_ALNUM = _BOLD_UPPER + _BOLD_LOWER + _BOLD_DIGITS
_BOLD_MAP = dict(zip(_PLAIN_ALNUM, _BOLD_ALNUM))
_BOLD_REV = {v: k for k, v in _BOLD_MAP.items()}


def to_bold(text: str) -> str:
    return "".join(_BOLD_MAP.get(ch, ch) for ch in text)


def from_bold(text: str) -> str:
    return "".join(_BOLD_REV.get(ch, ch) for ch in text)


# Button labels: plain English (for matching) + bold display text.
LBL_CONTACT = "Talk to Admin"
LBL_AI = "AI Chat"
LBL_HELP = "Help"
LBL_STOP = "Stop"
LBL_CANCEL = "Cancel"
LBL_BROADCAST = "Broadcast"
LBL_STATS = "Stats"
LBL_OWNER = "Owner"
LBL_LANG = "Language"
LBL_T2V = "Text to Voice"
LBL_T2V_ON = "Text to Voice: ON"
LBL_T2V_OFF = "Text to Voice: OFF"

BTN_CONTACT = to_bold(LBL_CONTACT)
BTN_AI = to_bold(LBL_AI)
BTN_HELP = to_bold(LBL_HELP)
BTN_STOP = to_bold(LBL_STOP)
BTN_CANCEL = to_bold(LBL_CANCEL)
BTN_BROADCAST = to_bold(LBL_BROADCAST)
BTN_STATS = to_bold(LBL_STATS)
BTN_OWNER = to_bold(LBL_OWNER)
BTN_LANG = to_bold(LBL_LANG)
BTN_T2V_ON = to_bold(LBL_T2V_ON)
BTN_T2V_OFF = to_bold(LBL_T2V_OFF)


def is_admin(user_id: int) -> bool:
    return bool(ADMIN_ID) and user_id == ADMIN_ID


def main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [[KeyboardButton(BTN_CONTACT, style="primary")],
         [KeyboardButton(BTN_AI, style="success"),
          KeyboardButton(BTN_HELP, style="primary")],
         [KeyboardButton(BTN_OWNER, style="danger"),
          KeyboardButton(BTN_LANG, style="primary")]],
        resize_keyboard=True)


def connected_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [[KeyboardButton(BTN_STOP, style="danger"),
          KeyboardButton(BTN_CANCEL, style="danger")]],
        resize_keyboard=True)


def admin_keyboard() -> ReplyKeyboardMarkup:
    # The Text to Voice button always shows the current state.
    t2v = BTN_T2V_ON if admin_voice_enabled() else BTN_T2V_OFF
    return ReplyKeyboardMarkup(
        [[KeyboardButton(BTN_BROADCAST, style="primary"),
          KeyboardButton(BTN_STATS, style="success")],
         [KeyboardButton(t2v, style="primary")]],
        resize_keyboard=True)


WELCOME_ADMIN = (
    "👑 <b>এডমিন প্যানেল</b>\n\n"
    "ইউজারের মেসেজ এভাবে আসবে:\n"
    "নাম / ইউজারনেম / চ্যাট আইডি / মেসেজ\n\n"
    "উত্তর দিতে সেই মেসেজে <b>রিপ্লাই</b> দিন —\n"
    "উত্তর সরাসরি সেই ইউজারের কাছে যাবে।\n"
    "অথবা ব্যবহার করুন: <code>/reply user_id মেসেজ</code>\n\n"
    "📢 ব্রডকাস্ট: <code>/broadcast মেসেজ</code>\n"
    "📊 পরিসংখ্যান: <code>/stats</code>\n\n"
    "🪪 মালিকের পরিচয় / তথ্য (শুধু আপনি):\n"
    "• <code>/setintro আপনার পরিচয়</code>\n"
    "• <code>/addinfo তথ্য</code> — তথ্য যোগ করুন\n"
    "• <code>/listinfo</code> — সব তথ্য দেখুন\n"
    "• <code>/delinfo নম্বর</code> — তথ্য মুছুন\n"
    "• <code>/owner</code> — পরিচয় দেখুন\n"
    "• <code>/unblock user_id</code> — স্প্যাম ব্লক তুলুন\n"
    "📊 পোল (শুধু আপনি):\n"
    "• <code>/poll প্রশ্ন</code> — সব ইউজারকে পোল পাঠান\n"
    "• <code>/pollresults</code> — পোলের ফলাফল দেখুন"
)

HELP_TEXT = (
    "ℹ️ <b>সাহায্য</b>\n\n"
    f"🤖 <b>{BTN_AI}</b> — প্রশ্ন লিখুন, অটো উত্তর পাবেন।\n"
    f"💬 <b>{BTN_CONTACT}</b> — চাপ দিলেই এডমিনের সাথে কানেক্ট হবেন।\n"
    "   একবার কানেক্ট হলে বারবার বোতাম চাপতে হবে না —\n"
    "   আপনার সব মেসেজ এডমিনের কাছে যাবে।\n"
    f"🛑 <b>{BTN_STOP} / {BTN_CANCEL}</b> — এডমিনের সাথে সংযোগ শেষ হবে,\n"
    "   তারপর থেকে আবার AI উত্তর দেবে।\n"
    f"👑 <b>{BTN_OWNER}</b> — বটের মালিকের পরিচয় দেখবেন।\n"
    f"🌐 <b>{BTN_LANG}</b> — বাংলা / English বদলান।"
)

HELP_TEXT_EN = (
    "ℹ️ <b>Help</b>\n\n"
    f"🤖 <b>{BTN_AI}</b> — write a question, get an auto answer.\n"
    f"💬 <b>{BTN_CONTACT}</b> — tap to connect with the admin.\n"
    "   Once connected, every message goes to the admin.\n"
    f"🛑 <b>{BTN_STOP} / {BTN_CANCEL}</b> — end the admin session,\n"
    "   then AI answers resume.\n"
    f"👑 <b>{BTN_OWNER}</b> — see the bot owner's identity.\n"
    f"🌐 <b>{BTN_LANG}</b> — switch between Bengali / English."
)


# ---------------- Full-bot translation (feature 14, extended) ----------------
# The Language button switches the WHOLE bot, not just AI answers:
# every user-facing message follows the user's own language choice
# (data/lang.json, bn/en). Admin-facing strings stay in Bengali
# (the admin is Abu). Reply-keyboard button labels keep their
# English bold-Unicode form (Bengali has no bold form).
STRINGS = {
    "welcome": {
        "bn": (
            "🔥 <b>স্বাগতম!</b>\n\n"
            "🤖 প্রশ্ন লিখুন — AI অটো উত্তর দেবে।\n"
            "{contact} চাপ দিলে এডমিনের সাথে কানেক্ট হবেন।\n"
            "একবার কানেক্ট হলে আপনার সব মেসেজ এডমিনের কাছে যাবে,\n"
            "বন্ধ করতে {stop} বা {cancel} চাপুন।\n"
            "{owner} চাপ দিলে বটের মালিকের পরিচয় দেখবেন।\n"
            "{lang} চাপ দিলে বাংলা / English বদলাতে পারবেন।"),
        "en": (
            "🔥 <b>Welcome!</b>\n\n"
            "🤖 Write a question — AI will auto answer.\n"
            "Tap {contact} to connect with the admin.\n"
            "Once connected, all your messages go to the admin,\n"
            "to stop tap {stop} or {cancel}.\n"
            "Tap {owner} to see the bot owner's identity.\n"
            "Tap {lang} to switch Bengali / English."),
    },
    "connected_start": {
        "bn": ("✅ আপনি এডমিনের সাথে কানেক্টেড আছেন।\n"
               "মেসেজ লিখলেই এডমিন পেয়ে যাবেন।"),
        "en": ("✅ You are connected with the admin.\n"
               "Just write your message and the admin will get it."),
    },
    "already_connected": {
        "bn": "আপনি ইতিমধ্যে এডমিনের সাথে কানেক্টেড আছেন। 🙂",
        "en": "You are already connected with the admin. 🙂",
    },
    "ai_on": {
        "bn": ("🤖 <b>AI চালু আছে!</b>\n\n"
               "আপনার প্রশ্নটি লিখে পাঠান, অটো উত্তর পাবেন।"),
        "en": ("🤖 <b>AI is on!</b>\n\n"
               "Write your question and you will get an auto answer."),
    },
    "ai_connected_hint": {
        "bn": "এডমিনের সাথে কানেক্টেড আছেন — AI চাইলে প্রথমে {stop} চাপুন।",
        "en": "You are connected with the admin — to use AI, tap {stop} first.",
    },
    "session_ended": {
        "bn": ("🔌 <b>এডমিনের সাথে সংযোগ শেষ।</b>\n\n"
               "এখন থেকে আবার AI উত্তর দেবে। 🙂"),
        "en": ("🔌 <b>Admin session ended.</b>\n\n"
               "AI will answer you again from now on. 🙂"),
    },
    "not_connected": {
        "bn": ("আপনি এডমিনের সাথে কানেক্টেড নন। 🙂\n"
               "এডমিনের সাথে কথা বলতে {contact} চাপ দিন।"),
        "en": ("You are not connected with the admin. 🙂\n"
               "To talk to the admin, tap {contact}."),
    },
    "voice_not_understood": {
        "bn": "ভয়েস বুঝতে পারিনি 🙂 লিখে পাঠালে উত্তর দেব।",
        "en": "I couldn't understand the voice 🙂 Please write it and I will answer.",
    },
    "photo_not_understood": {
        "bn": "ছবিটা দেখতে পারলাম না 🙂 লিখে জানালে উত্তর দেব।",
        "en": "I couldn't see the photo 🙂 Please write it and I will answer.",
    },
    "media_fallback": {
        "bn": ("ছবি/ফাইল দেখতে পারছি না 🙂 লিখে জানালে উত্তর দেব।\n"
               "এডমিনকে পাঠাতে চাইলে {contact} চাপ দিন।"),
        "en": ("I can't see photos/files 🙂 Please write it and I will answer.\n"
               "To reach the admin, tap {contact}."),
    },
    "send_failed": {
        "bn": "❌ মেসেজ পাঠানো যায়নি। কিছুক্ষণ পর আবার চেষ্টা করুন।",
        "en": "❌ The message could not be sent. Please try again in a little while.",
    },
    "thanks": {
        "bn": "ধন্যবাদ! 🙂",
        "en": "Thank you! 🙂",
    },
    "sorry_admin": {
        "bn": "দুঃখিত! এডমিন দেখছেন।",
        "en": "Sorry! The admin is looking into it.",
    },
    "connect_confirm": {
        "bn": ("✅ <b>এডমিনের সাথে কানেক্ট হয়েছেন!</b>\n\n"
               "এখন যা লিখবেন সব সরাসরি এডমিনের কাছে যাবে।\n"
               "বন্ধ করতে {stop} বা {cancel} চাপুন।"),
        "en": ("✅ <b>You are connected with the admin!</b>\n\n"
               "Everything you write now goes straight to the admin.\n"
               "To stop, tap {stop} or {cancel}."),
    },
    "error_retry": {
        "bn": "⚠️ একটু সমস্যা হয়েছে, আবার চেষ্টা করুন।",
        "en": "⚠️ Something went wrong, please try again.",
    },
    "spam_blocked": {
        "bn": ("⛔ অতিরিক্ত মেসেজের কারণে আপনাকে ব্লক করা হয়েছে।\n"
               "এডমিনের সাথে যোগাযোগ করুন।"),
        "en": ("⛔ You have been blocked for sending too many messages.\n"
               "Please contact the admin."),
    },
    "not_sure": {
        "bn": ("এই বিষয়ে আমি নিশ্চিত নই — সঠিক তথ্যের জন্য এডমিনকে "
               "জিজ্ঞেস করুন, {contact} চাপ দিন। 🙂"),
        "en": ("I'm not sure about this one — please ask the admin for "
               "the exact answer, tap {contact}. 🙂"),
    },
}


def t(key: str, lang: str = "bn") -> str:
    """One user-facing string in the user's language.

    {contact}, {ai}, {stop}, {cancel}, {owner}, {lang} and {help}
    are filled with the bold button labels at call time.
    """
    text = STRINGS[key]["en" if lang == "en" else "bn"]
    return text.format(contact=BTN_CONTACT, ai=BTN_AI, stop=BTN_STOP,
                       cancel=BTN_CANCEL, owner=BTN_OWNER,
                       lang=BTN_LANG, help=BTN_HELP)


def help_text(lang: str = "bn") -> str:
    return HELP_TEXT_EN if lang == "en" else HELP_TEXT


def offline_msg(lang: str = "bn") -> str:
    if lang == "en":
        return ("The admin is offline now, you can talk in the "
                "morning. 🙂\nIf you have a question, write it — "
                "AI will answer.")
    return OFFLINE_MSG


# ---------------- AI auto-reply ----------------

def rule_reply(text: str, lang: str = "bn") -> str:
    t = text.lower()
    if lang == "en":
        if any(w in t for w in ["hello", "hi", "salam", "assalam"]):
            return ("Hello! 👋 How are you?\n"
                    "Write your question and I will try to answer.")
        if any(w in t for w in ["thank"]):
            return "You're welcome! 😊 Ask me anything else."
        if any(w in t for w in ["price", "cost", "how much"]):
            return ("For prices and offers, please talk to the admin —\n"
                    f"tap {BTN_CONTACT} and the admin will explain.")
        if any(w in t for w in ["how are you"]):
            return "I'm doing well, thank you! 😊 How are you?"
        if any(w in t for w in ["bye"]):
            return "Goodbye! Come back anytime. 👋"
        if any(w in t for w in ["help"]):
            return help_text(lang)
        return ("I didn't understand, could you write a bit more? 🙂\n"
                f"To talk to the admin, tap {BTN_CONTACT}\n"
                "or write \"admin\" in your message.")
    if any(w in t for w in ["সালাম", "আসসালাম", "hello", "hi", "হ্যালো", "হাই"]):
        return ("ওয়ালাইকুম সালাম! 👋 কেমন আছেন?\n"
                "আপনার প্রশ্নটি লিখুন, আমি উত্তর দেওয়ার চেষ্টা করব।")
    if any(w in t for w in ["ধন্যবাদ", "thanks", "thank"]):
        return "আপনাকে স্বাগতম! 😊 আর কিছু জানতে চাইলে লিখুন।"
    if any(w in t for w in ["দাম", "price", "কত", "টাকা", "খরচ"]):
        return ("দাম/অফার সম্পর্কে জানতে এডমিনের সাথে কথা বলুন —\n"
                f"{BTN_CONTACT} চাপ দিন, এডমিন বিস্তারিত জানিয়ে দেবেন।")
    if any(w in t for w in ["কেমন আছ", "ভালো আছ"]):
        return "আমি ভালো আছি, ধন্যবাদ! 😊 আপনি কেমন আছেন?"
    if any(w in t for w in OWNER_WORDS):
        return owner_text(lang)
    if any(w in t for w in ["তুমি কে", "তোমার নাম", "who are you"]):
        return ("আমি এই বটের অটো সহকারী। 🤖\n"
                "সাধারণ প্রশ্নের উত্তর দিতে পারি; জটিল বিষয়ে এডমিনের "
                "সাথে কথা বলুন।")
    if any(w in t for w in ["বাই", "bye", "আল্লাহ হাফেজ"]):
        return "আল্লাহ হাফেজ! আবার আসবেন। 👋"
    if any(w in t for w in ["সাহায্য", "help", "হেল্প"]):
        return help_text(lang)
    return ("বুঝতে পারিনি, আরেকটু বিস্তারিত লিখবেন? 🙂\n"
            f"এডমিনের সাথে কথা বলতে চাইলে {BTN_CONTACT} চাপ দিন\n"
            "অথবা মেসেজে “এডমিন” লিখুন।")


def gemini_reply(text: str, history: list | None = None,
                 lang: str = "bn") -> str | None:
    """Real AI reply via Gemini. Returns None on any failure."""
    if not GEMINI_KEY:
        return None
    try:
        import json as _json
        url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               "gemini-2.0-flash:generateContent?key=" + GEMINI_KEY)
        reply_lang = ("Reply in Bengali" if lang == "bn"
                      else "Reply in English")
        system = ("You are a friendly support assistant for a "
                  "Telegram bot. " + reply_lang + ", short (1-3 lines), "
                  "warm and helpful. Never answer with just "
                  "'I don't know'. Always give the closest helpful "
                  "answer you can; if you really cannot answer, say "
                  "briefly that you will ask the admin. "
                  "If the user asks for a human/admin, "
                  "tell them to tap the admin button. "
                  + owner_context() + style_context())
        contents = []
        for h in (history or []):
            contents.append({"role": "user" if h["role"] == "user"
                             else "model",
                             "parts": [{"text": h["text"]}]})
        contents.append({"role": "user", "parts": [{"text": text}]})
        body = _json.dumps({
            "system_instruction": {"parts": [{"text": system}]},
            "contents": contents,
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        log.warning("gemini reply failed: %s", e)
        return None


def pollinations_reply(text: str, history: list | None = None,
                       lang: str = "bn") -> str | None:
    """Real AI reply via the free Pollinations text API (no key).

    Tested 2026-11-22: with temperature=0 and a short English
    system prompt it answered 3/3 basic factual questions
    correctly (Dhaka, sunrise east, H2O). The earlier failures
    came from a long Bengali instruction + default temperature.
    2026-11-22 afternoon: the GET form (whole prompt in the URL)
    timed out twice even at 60s — the POST form with a JSON body
    answered the same question in ~18s. So this uses POST now.
    Returns None on any failure so the rule-based reply can cover.
    """
    try:
        import json
        reply_lang = ("Answer briefly in Bengali" if lang == "bn"
                      else "Answer briefly in English")
        system = ("You are the auto assistant of this Telegram bot. "
                  "Never call yourself ChatGPT and never say you were "
                  "made by OpenAI. " + reply_lang + ", in 1-3 "
                  "lines. Never answer with just 'I don't know'. "
                  "Always give the closest helpful answer you can; "
                  "if you really cannot answer, say briefly that you "
                  "will ask the admin and tell the user to tap the "
                  "Talk to Admin button. "
                  + owner_context() + style_context())
        messages = [{"role": "system", "content": system}]
        for h in (history or [])[-10:]:
            role = "user" if h["role"] == "user" else "assistant"
            messages.append({"role": role, "content": h["text"][:500]})
        messages.append({"role": "user", "content": text})
        body = json.dumps({"messages": messages, "model": "openai",
                           "temperature": 0}).encode("utf-8")
        req = urllib.request.Request(
            "https://text.pollinations.ai/", data=body,
            headers={"Content-Type": "application/json",
                     "Accept": "text/plain"})
        answer = None
        for _attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    answer = resp.read().decode("utf-8", "ignore").strip()
                break
            except Exception as e:
                log.warning("pollinations attempt failed: %s", e)
        if not answer:
            return None
        return answer[:1500]
    except Exception as e:
        log.warning("pollinations reply failed: %s", e)
        return None


def openrouter_reply(text: str, history: list | None = None,
                     lang: str = "bn") -> str | None:
    """Real AI reply via OpenRouter, Groq-served Llama 3.3 70B.

    Same system prompt, history and reply style as the other
    backends (Bengali or English per the user's language choice).
    The API key comes from the registered custom.openrouter
    connection through the authd surrogate exchange — never from a
    file, env var or log. Returns None on any failure so the chain
    falls back to Gemini / Pollinations / rule replies.
    """
    if not OPENROUTER_AVAILABLE:
        return None
    try:
        import json
        import urllib.error
        reply_lang = ("Answer briefly in Bengali" if lang == "bn"
                      else "Answer briefly in English")
        system = ("You are the auto assistant of this Telegram bot. "
                  "Never call yourself ChatGPT and never say you were "
                  "made by OpenAI. " + reply_lang + ", in 1-3 "
                  "lines. Never answer with just 'I don't know'. "
                  "Always give the closest helpful answer you can; "
                  "if you really cannot answer, say briefly that you "
                  "will ask the admin and tell the user to tap the "
                  "Talk to Admin button. "
                  + owner_context() + style_context())
        messages = [{"role": "system", "content": system}]
        for h in (history or [])[-10:]:
            role = "user" if h["role"] == "user" else "assistant"
            messages.append({"role": role, "content": h["text"][:500]})
        messages.append({"role": "user", "content": text})
        body = json.dumps({
            "model": OPENROUTER_MODEL,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 600,
            "provider": {"order": ["groq"], "allow_fallbacks": False},
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        add_surrogate_to_request(req, "custom.openrouter",
                                 allowed_hosts=["openrouter.ai"])
        answer = None
        for _attempt in range(2):
            try:
                with _OPENROUTER_LOCK:
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                answer = data["choices"][0]["message"]["content"].strip()
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "ignore")[:300]
                log.warning("openrouter attempt failed: HTTP %s %s",
                            e.code, detail)
                if e.code == 402 and _attempt == 0:
                    # In-flight budget exhausted: wait for the other
                    # requests to settle, then try once more.
                    time.sleep(3)
                    continue
                break
            except Exception as e:
                log.warning("openrouter attempt failed: %s", e)
        if not answer:
            return None
        return answer[:1500]
    except Exception as e:
        log.warning("openrouter reply failed: %s", e)
        return None


# Ignorance-only answers ("I don't know") must never reach the user
# (Abu, 2026-11-24): the old system prompts told the models to say
# exactly that, and the small Pollinations model obeyed literally.
_IGNORANCE_PREFIXES = ("i don't know", "i do not know", "i dont know",
                       "আমি জানি না", "আমি জানিনা")


def _is_ignorance(ans: str) -> bool:
    """True when a reply is (almost) only an 'I don't know'.

    Only short replies count: a longer answer that starts with
    "I don't know" but then gives guidance is a real answer.
    Note: no aggressive character cleaning here — stripping
    non-word characters would also strip Bengali vowel signs
    (they are combining marks, not letters) and mangle the text.
    """
    raw = (ans or "").strip().lower()
    raw = re.sub(r"^[^\w]+", "", raw)  # strip leading emoji/punct
    if not raw or len(raw) > 80:
        return False
    if any(raw.startswith(p) for p in _IGNORANCE_PREFIXES):
        return True
    return "don't know" in raw or "do not know" in raw


def ai_reply(text: str, history: list | None = None,
             lang: str = "bn") -> str:
    saw_ignorance = False
    for name, fn in (("openrouter/groq", openrouter_reply),
                     ("gemini", gemini_reply),
                     ("pollinations", pollinations_reply)):
        ans = fn(text, history, lang)
        if ans:
            if _is_ignorance(ans):
                log.info("ai reply via %s was ignorance-only; "
                         "trying next backend", name)
                saw_ignorance = True
                continue
            log.info("ai reply via %s", name)
            return ans
    if saw_ignorance:
        # Every backend that answered only said it does not know:
        # point the user at the admin instead of repeating that.
        return t("not_sure", lang)
    return rule_reply(text, lang)


# ---------------- Voice messages (feature 3) ----------------
# Telegram voice notes are Opus-in-Ogg; they are converted to 16kHz
# mono WAV with ffmpeg, then transcribed via OpenRouter
# gpt-audio-mini through the chat/completions endpoint (the same
# proven path as openrouter_reply). No whisper model is listed on
# OpenRouter and /audio/transcriptions is not reachable from this
# host, so audio goes through the chat API instead.
TRANSCRIBE_MODEL = "openai/gpt-audio-mini"


def transcribe_audio(wav_bytes: bytes) -> str | None:
    """Transcribe Bengali voice audio. Returns None on any failure."""
    if not OPENROUTER_AVAILABLE:
        return None
    try:
        import json
        b64 = base64.b64encode(wav_bytes).decode("ascii")
        body = json.dumps({
            "model": TRANSCRIBE_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": "Transcribe this Bengali audio exactly, "
                             "word for word. Reply with only the "
                             "transcript."},
                    {"type": "input_audio",
                     "input_audio": {"data": b64, "format": "wav"}},
                ],
            }],
            "temperature": 0,
            "max_tokens": 600,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        add_surrogate_to_request(req, "custom.openrouter",
                                 allowed_hosts=["openrouter.ai"])
        with _OPENROUTER_LOCK:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"].strip()
        return text or None
    except Exception as e:
        log.warning("transcribe failed: %s", e)
        return None


async def transcribe_voice_message(context: ContextTypes.DEFAULT_TYPE,
                                   msg) -> str | None:
    """Download a Telegram voice note, convert to WAV, transcribe."""
    tmp = BASE / "tmp"
    tmp.mkdir(exist_ok=True)
    stamp = int(time.time() * 1000)
    oga = tmp / f"voice_{stamp}.oga"
    wav = tmp / f"voice_{stamp}.wav"
    try:
        tg_file = await context.bot.get_file(msg.voice.file_id)
        await tg_file.download_to_drive(oga)
        r = await asyncio.to_thread(
            subprocess.run,
            ["ffmpeg", "-y", "-v", "error", "-i", str(oga),
             "-ar", "16000", "-ac", "1", str(wav)],
            capture_output=True, timeout=60)
        if r.returncode != 0 or not wav.exists():
            log.warning("voice convert failed rc=%s", r.returncode)
            return None
        text = await asyncio.to_thread(transcribe_audio, wav.read_bytes())
        if text:
            log.info("voice transcribed via %s", TRANSCRIBE_MODEL)
        return text
    except Exception as e:
        log.warning("voice transcribe failed: %s", e)
        return None
    finally:
        for p in (oga, wav):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


# ---------------- Photo understanding / vision (feature 7) ----------------
# A photo sent in AI mode is described/answered by a vision model on
# OpenRouter. Qwen2.5-VL 72B first (proven live 2026-11-24); Llama 4
# Scout is listed in /models but its chat call returns 404, so it is
# only a fallback. NO Google/Gemini model is used anywhere
# (Abu refused Gemini).
VISION_MODELS = (
    ("qwen/qwen2.5-vl-72b-instruct", None),
    ("meta-llama/llama-4-scout",
     {"order": ["groq"], "allow_fallbacks": False}),
)


def vision_reply(image_bytes: bytes, question: str,
                 lang: str = "bn") -> str | None:
    """Answer about a photo. Returns None if no vision model works."""
    if not OPENROUTER_AVAILABLE:
        return None
    import json
    b64 = base64.b64encode(image_bytes).decode("ascii")
    reply_lang = ("Answer in Bengali" if lang == "bn"
                  else "Answer in English")
    prompt = ("Look at this image. " + reply_lang + ", briefly "
              "(1-3 lines). "
              + ("The user asks: " + question if question
                 else "Describe what the image shows."))
    for model, provider in VISION_MODELS:
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": "data:image/jpeg;base64,"
                                   + b64}},
                ]}],
                "temperature": 0,
                "max_tokens": 600,
            }
            if provider:
                payload["provider"] = provider
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=body, headers={"Content-Type": "application/json"})
            add_surrogate_to_request(req, "custom.openrouter",
                                     allowed_hosts=["openrouter.ai"])
            with _OPENROUTER_LOCK:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            ans = data["choices"][0]["message"]["content"].strip()
            if ans:
                log.info("vision reply via %s", model)
                return ans[:1500]
        except Exception as e:
            log.warning("vision via %s failed: %s", model, e)
    return None


async def describe_photo_message(context: ContextTypes.DEFAULT_TYPE,
                                 msg, lang: str = "bn") -> str | None:
    """Download a Telegram photo (largest size) and describe it."""
    tmp = BASE / "tmp"
    tmp.mkdir(exist_ok=True)
    stamp = int(time.time() * 1000)
    jpg = tmp / f"photo_{stamp}.jpg"
    try:
        tg_file = await context.bot.get_file(msg.photo[-1].file_id)
        await tg_file.download_to_drive(jpg)
        data = jpg.read_bytes()
        if len(data) > 1_500_000:
            # Downscale large photos so the base64 payload stays sane.
            small = tmp / f"photo_{stamp}_small.jpg"
            r = await asyncio.to_thread(
                subprocess.run,
                ["ffmpeg", "-y", "-v", "error", "-i", str(jpg),
                 "-vf", "scale=1024:-1", str(small)],
                capture_output=True, timeout=60)
            if r.returncode == 0 and small.exists():
                data = small.read_bytes()
                small.unlink(missing_ok=True)
        question = (msg.caption or "").strip()
        return await asyncio.to_thread(vision_reply, data, question,
                                       lang)
    except Exception as e:
        log.warning("photo describe failed: %s", e)
        return None
    finally:
        try:
            jpg.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------- Rating under AI replies (feature 9) ----------------
# Every AI answer carries inline 👍 / 👎 buttons. Ratings are stored
# in data/ratings.json; a 👎 also forwards the exchange to the admin
# for follow-up. (Inline buttons are used ONLY here; the menus stay
# ReplyKeyboardMarkup, per the original design.)
RATINGS_FILE = BASE / "data" / "ratings.json"
_RATING_PENDING: dict = {}  # token -> {user_id, question, answer}

# Admin-message translate store: token -> {text, user_id, full,
# to_lang, translated, shown}. In-memory only; a restart clears it,
# so old buttons just answer that the message is gone.
_ADMIN_MSG: dict = {}


def detect_msg_lang(text: str) -> str:
    """'bn' if the text contains Bengali script, else 'en'."""
    return "bn" if re.search(r"[\u0980-\u09FF]", text or "") else "en"


def _admin_msg_markup(token: str, entry: dict) -> InlineKeyboardMarkup:
    """Translate toggle button for one admin message."""
    if entry["shown"]:
        label = to_bold("Hide translation")
    elif entry["to_lang"] == "en":
        label = to_bold("Translate to English")
    else:
        label = to_bold("Translate to Bangla")
    return InlineKeyboardMarkup([[
        InlineKeyboardButton(label, callback_data=f"tr:{token}",
                             style="primary"),
    ]])


def translate_text(text: str, target_lang: str) -> str | None:
    """Translate a user message for the admin (Groq first,
    Pollinations fallback). Short translate-only prompt, NOT a full
    AI answer. Returns None on any failure."""
    target = "English" if target_lang == "en" else "Bengali"
    system = (f"You are a translator. Translate the following message "
              f"to {target}. Reply with ONLY the translation, no "
              "explanation, no quotes, no extra text.")
    clipped = (text or "")[:1000]
    if not clipped.strip():
        return None
    if OPENROUTER_AVAILABLE:
        try:
            import json
            import urllib.error
            body = json.dumps({
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": clipped}],
                "temperature": 0,
                "max_tokens": 600,
                "provider": {"order": ["groq"], "allow_fallbacks": False},
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions", data=body,
                headers={"Content-Type": "application/json"})
            add_surrogate_to_request(req, "custom.openrouter",
                                     allowed_hosts=["openrouter.ai"])
            with _OPENROUTER_LOCK:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            ans = data["choices"][0]["message"]["content"].strip()
            if ans:
                return ans[:1500]
        except Exception as e:
            log.warning("translate via openrouter failed: %s", e)
    try:
        import json
        body = json.dumps({
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": clipped}],
            "model": "openai", "temperature": 0}).encode("utf-8")
        req = urllib.request.Request(
            "https://text.pollinations.ai/", data=body,
            headers={"Content-Type": "application/json",
                     "Accept": "text/plain"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            ans = resp.read().decode("utf-8", "ignore").strip()
        return ans[:1500] if ans else None
    except Exception as e:
        log.warning("translate via pollinations failed: %s", e)
        return None


def load_ratings() -> list:
    if RATINGS_FILE.exists():
        try:
            items = json.loads(RATINGS_FILE.read_text(encoding="utf-8"))
            return items if isinstance(items, list) else []
        except Exception:
            pass
    return []


def save_rating(entry: dict) -> None:
    items = load_ratings()
    items.append(entry)
    RATINGS_FILE.write_text(json.dumps(items, ensure_ascii=False,
                                       indent=2), encoding="utf-8")


async def rate_callback(update: Update,
                        context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data:
        return
    try:
        _, verdict, token = query.data.split(":", 2)
    except ValueError:
        await query.answer()
        return
    entry = _RATING_PENDING.get(token)
    _lang = get_lang(entry["user_id"]) if entry else "bn"
    await query.answer(t("thanks", _lang) if verdict == "good"
                       else t("sorry_admin", _lang))
    try:
        await query.edit_message_reply_markup(reply_markup=None)
    except Exception:
        pass
    if not entry:
        return
    try:
        save_rating({"user_id": entry["user_id"],
                     "question": entry["question"],
                     "answer": entry["answer"],
                     "rating": verdict, "ts": int(time.time())})
    except Exception as e:
        log.warning("saving rating failed: %s", e)
    if verdict == "bad":
        try:
            await context.bot.send_message(
                chat_id=ADMIN_ID,
                text=("👎 <b>খারাপ রেটিং</b>\n\n"
                      f"ইউজার: {entry['user_id']}\n"
                      f"প্রশ্ন: {entry['question'][:300]}\n"
                      f"AI উত্তর: {entry['answer'][:500]}"),
                parse_mode=ParseMode.HTML)
            log.info("bad rating forwarded for user %s",
                     entry["user_id"])
        except Exception as e:
            log.warning("bad-rating forward failed: %s", e)
    _RATING_PENDING.pop(token, None)


async def lang_callback(update: Update,
                        context: ContextTypes.DEFAULT_TYPE) -> None:
    """Feature 14 (corrected): language chosen by inline button tap."""
    query = update.callback_query
    if not query or not query.data:
        return
    try:
        _, lang = query.data.split(":", 1)
    except ValueError:
        await query.answer()
        return
    if lang not in ("bn", "en"):
        await query.answer()
        return
    user = query.from_user
    set_lang(user.id, lang)
    if lang == "en":
        text = ("Language set to English. 🇬🇧\n"
                "The whole bot will now speak English.")
        toast = "English selected"
    else:
        text = ("ভাষা বেছে নেওয়া হয়েছে: বাংলা। 🇧🇩\n"
                "এখন থেকে পুরো বট বাংলায় কথা বলবে।")
        toast = "বাংলা বেছে নেওয়া হয়েছে"
    await query.answer(toast)
    try:
        await query.edit_message_text(text)
    except Exception:
        try:
            await query.message.reply_text(text)
        except Exception:
            pass
    log.info("user %s selected language %s", user.id, lang)


async def translate_callback(update: Update,
                             context: ContextTypes.DEFAULT_TYPE) -> None:
    """Admin Translate button: toggle the translation of one user
    message on the admin's copy. The original text always stays."""
    query = update.callback_query
    if not query or not query.data:
        return
    try:
        _, token = query.data.split(":", 1)
    except ValueError:
        await query.answer()
        return
    entry = _ADMIN_MSG.get(token)
    if not entry:
        await query.answer("মেসেজটি আর নেই।")
        return
    if not entry["shown"]:
        if not entry["translated"]:
            translated = await asyncio.to_thread(
                translate_text, entry["text"], entry["to_lang"])
            if not translated:
                await query.answer("অনুবাদ করা যায়নি।")
                return
            entry["translated"] = translated
        target = "English" if entry["to_lang"] == "en" else "বাংলা"
        new_text = (entry["full"] + f"\n\n🌐 অনুবাদ ({target}):\n"
                    + entry["translated"])
        entry["shown"] = True
        await query.answer("অনুবাদ দেখানো হয়েছে।")
    else:
        new_text = entry["full"]
        entry["shown"] = False
        await query.answer("মূল মেসেজ দেখানো হয়েছে।")
    try:
        await query.edit_message_text(
            new_text, reply_markup=_admin_msg_markup(token, entry))
    except Exception as e:
        log.warning("translate edit failed: %s", e)


async def send_ai_answer(msg, user, question: str, answer: str,
                         lang: str = "bn") -> None:
    """Send an AI answer with 👍/👎 rating buttons, record history.

    AI replies are ALWAYS text (Abu, 2026-11-24) — never voice,
    even when the user sent a voice message.
    """
    token = f"{user.id}:{int(time.time())}"
    _RATING_PENDING[token] = {"user_id": user.id,
                              "question": question[:500],
                              "answer": answer[:1000]}
    markup = InlineKeyboardMarkup([[
        InlineKeyboardButton("👍", callback_data=f"rate:good:{token}",
                             style="success"),
        InlineKeyboardButton("👎", callback_data=f"rate:bad:{token}",
                             style="danger"),
    ]])
    await msg.reply_text(answer, parse_mode=ParseMode.HTML,
                         reply_markup=markup)
    add_history(user.id, "user", question)
    add_history(user.id, "bot", plain_text(answer))
    log.info("ai reply to user %s", user.id)
    # AI voice replies are disabled (Abu, 2026-11-24): the answer
    # above is text-only, even for voice-initiated chats. Voice
    # output now exists only for the admin's Text-to-Voice toggle
    # (send_admin_reply_as_voice), never for the AI.


# ---------------- Voice replies / TTS (admin toggle only) ----------------
# TTS is used ONLY for the admin's Text-to-Voice toggle
# (send_admin_reply_as_voice). The AI never sends voice replies
# (Abu, 2026-11-24). Voice comes FIRST from Abu's own Cartesia
# voice clone (he made it himself; account holds an owned Bengali
# clone named "Abu Taher", verified 2026-11-24) and falls back to
# the local tts binary (Meta Shortwave, --language bn) if Cartesia
# is unavailable. The Cartesia WAV has no ID3 tag; the local binary
# writes a WAV with a leading ID3 tag that ffmpeg's wav demuxer
# rejects, so the tag is stripped first, then ffmpeg converts to
# Opus/OGG for Telegram send_voice.
TTS_BIN = "/opt/hatch/bin/tts"
TTS_VOICE = "avocado_v2:MAI_03"
# Abu's Cartesia clone voice id (an identifier, not a secret). The
# API key comes from the registered custom.cartesia connection
# through the authd surrogate exchange, never from a file.
CARTESIA_VOICE_ID = "fbad9174-a690-4db2-a82d-076c7c45f927"
CARTESIA_MODEL = "sonic-3"
# Same dynamic_credentials import as OpenRouter; the flag only says
# the shared import worked.
CARTESIA_AVAILABLE = OPENROUTER_AVAILABLE


def make_cartesia_wav(text: str, lang: str = "bn") -> bytes | None:
    """Bengali/English TTS via Abu's Cartesia clone voice.

    Returns WAV bytes (pcm_s16le, 44100 Hz, mono) or None on any
    failure so the caller falls back to the local tts binary.
    """
    if not CARTESIA_AVAILABLE:
        return None
    try:
        import json
        import urllib.error
        body = json.dumps({
            "model_id": CARTESIA_MODEL,
            "transcript": text,
            "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
            "language": "bn" if lang == "bn" else "en",
            "output_format": {
                "container": "wav",
                "encoding": "pcm_s16le",
                "sample_rate": 44100,
            },
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.cartesia.ai/tts/bytes", data=body,
            headers={"Content-Type": "application/json",
                     "Cartesia-Version": "2026-08-14"})
        add_surrogate_to_request(req, "custom.cartesia",
                                 allowed_hosts=["api.cartesia.ai"])
        with urllib.request.urlopen(req, timeout=120) as resp:
            audio = resp.read()
        if not audio or len(audio) < 1000:
            log.warning("cartesia tts returned too little audio")
            return None
        return audio
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:300]
        log.warning("cartesia tts failed: HTTP %s %s", e.code, detail)
        return None
    except Exception as e:
        log.warning("cartesia tts failed: %s", e)
        return None


def make_voice_bytes(text: str, lang: str = "bn") -> bytes | None:
    """Bengali/English TTS -> Opus/OGG bytes. Returns None on failure."""
    clean = re.sub(r"<[^>]+>", "", text or "").strip()
    clean = clean[:600]
    if not clean:
        return None
    tmp = BASE / "tmp"
    tmp.mkdir(exist_ok=True)
    stamp = int(time.time() * 1000)
    wav = tmp / f"tts_{stamp}.wav"
    ogg = tmp / f"tts_{stamp}.ogg"
    try:
        cartesia_wav = (make_cartesia_wav(clean, lang)
                        if CARTESIA_AVAILABLE else None)
        if cartesia_wav:
            wav.write_bytes(cartesia_wav)
            log.info("voice via cartesia clone")
        else:
            r = subprocess.run(
                [TTS_BIN, "speak", "--language",
                 "bn" if lang == "bn" else "en", "--format", "wav",
                 "--output", str(wav), "--text-stdin"],
                input=clean.encode("utf-8"), capture_output=True,
                timeout=120)
            if r.returncode != 0 or not wav.exists():
                log.warning("tts failed rc=%s", r.returncode)
                return None
        raw = wav.read_bytes()
        i = raw.find(b"RIFF")
        if i > 0:
            raw = raw[i:]
            wav.write_bytes(raw)
        r2 = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(wav),
             "-c:a", "libopus", "-b:a", "64k", str(ogg)],
            capture_output=True, timeout=60)
        if r2.returncode != 0 or not ogg.exists():
            log.warning("tts ogg convert failed rc=%s", r2.returncode)
            return None
        return ogg.read_bytes()
    except Exception as e:
        log.warning("tts failed: %s", e)
        return None
    finally:
        for p in (wav, ogg):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


async def send_admin_reply_as_voice(context, target_id: int,
                                    text: str) -> bool:
    """Requirement 3 (corrected): deliver the admin's typed reply to
    the user as a VOICE message (same TTS path as feature 10),
    following the reply's language. Falls back to sending the text
    if TTS fails. Returns True if voice was sent."""
    lang = detect_msg_lang(text)
    data = None
    try:
        data = await asyncio.to_thread(make_voice_bytes, text, lang)
    except Exception as e:
        log.warning("admin reply tts failed: %s", e)
    if data:
        try:
            bio = io.BytesIO(data)
            bio.name = "voice.ogg"
            await context.bot.send_voice(chat_id=target_id, voice=bio)
            log.info("admin reply sent as voice to user %s", target_id)
            return True
        except Exception as e:
            log.warning("admin reply voice send failed: %s", e)
    await context.bot.send_message(chat_id=target_id, text=text)
    log.info("admin reply voice failed, sent as text to user %s",
             target_id)
    return False


async def send_admin_reply(context, target_id: int,
                           text: str) -> bool:
    """Deliver the admin's typed reply, gated by the Text-to-Voice
    toggle: voice when ON (TTS, text fallback on failure), plain
    text when OFF. Returns True only if a voice message was sent."""
    if admin_voice_enabled():
        return await send_admin_reply_as_voice(context, target_id, text)
    await context.bot.send_message(chat_id=target_id, text=text)
    log.info("admin reply sent as text to user %s", target_id)
    return False


# ---------------- sending to admin (never forwarded) ----------------

def admin_header(user) -> str:
    username = f"@{user.username}" if user.username else "নেই"
    return (
        f"নাম: {user.first_name}\n"
        f"ইউজারনেম: {username}\n"
        f"চ্যাট আইডি: {user.id}"
    )


async def urgent_alert(context: ContextTypes.DEFAULT_TYPE,
                       user, text: str) -> None:
    """Feature 8: immediate separate alert to the admin for urgent
    messages, in addition to the normal admin message / AI reply."""
    try:
        await context.bot.send_message(
            chat_id=ADMIN_ID,
            text=("🚨 <b>জরুরি মেসেজ!</b>\n\n"
                  f"নাম: {user.first_name}\n"
                  f"চ্যাট আইডি: {user.id}\n"
                  f"মেসেজ👇\n{(text or '')[:500]}"),
            parse_mode=ParseMode.HTML)
        log.info("urgent alert sent for user %s", user.id)
    except Exception as e:
        log.warning("urgent alert failed: %s", e)


async def send_to_admin(update: Update, context: ContextTypes.DEFAULT_TYPE,
                        msg) -> None:
    """Send the user's message to the admin as plain text (no forward).

    Stores admin-message-id -> user-id so the admin's direct reply
    to it can be routed back to the user.
    """
    user = update.effective_user
    DATA["users"].setdefault(str(user.id), {
        "name": user.full_name,
        "username": user.username or "",
    })
    header = admin_header(user)
    text = (msg.text or msg.caption or "").strip()
    try:
        sent_ids = []
        if msg.photo:
            caption = header + "\nমেসেজ👇\n" + (text or "📷 ছবি পাঠিয়েছে")
            sent = await context.bot.send_photo(
                chat_id=ADMIN_ID, photo=msg.photo[-1].file_id,
                caption=caption[:1000])
            sent_ids.append(sent.message_id)
        elif text:
            token = f"{user.id}:{int(time.time())}"
            full = header + "\nমেসেজ👇\n" + text
            to_lang = "en" if detect_msg_lang(text) == "bn" else "bn"
            if len(_ADMIN_MSG) > 500:  # keep the in-memory store bounded
                for _k in list(_ADMIN_MSG)[:len(_ADMIN_MSG) - 500]:
                    _ADMIN_MSG.pop(_k, None)
            _ADMIN_MSG[token] = {"text": text, "user_id": user.id,
                                 "full": full, "to_lang": to_lang,
                                 "translated": None, "shown": False}
            sent = await context.bot.send_message(
                chat_id=ADMIN_ID, text=full,
                reply_markup=_admin_msg_markup(token, _ADMIN_MSG[token]))
            sent_ids.append(sent.message_id)
        else:
            sent = await context.bot.send_message(
                chat_id=ADMIN_ID,
                text=header + "\nমেসেজ👇\n(ছবি/ফাইল/স্টিকার পাঠিয়েছে)")
            sent_ids.append(sent.message_id)
            copied = await context.bot.copy_message(
                chat_id=ADMIN_ID, from_chat_id=msg.chat_id,
                message_id=msg.message_id)
            sent_ids.append(copied.message_id)
        threads = DATA.setdefault("threads", {})
        for mid in sent_ids:
            threads[str(mid)] = user.id
        DATA["forwarded"] = DATA.get("forwarded", 0) + 1
        save_data(DATA)
        log.info("sent message from user %s to admin (admin_msg_ids=%s)",
                 user.id, sent_ids)
        if is_urgent(text):
            await urgent_alert(context, user, text)
    except Exception as e:
        log.error("send to admin failed for user %s: %s", user.id, e)
        await msg.reply_text(
            t("send_failed", get_lang(user.id)))


async def connect_user(update: Update, context: ContextTypes.DEFAULT_TYPE,
                       msg, user) -> None:
    """Put the user into a live admin session and confirm it."""
    CONNECTED.add(user.id)
    lang = get_lang(user.id)
    await msg.reply_text(
        t("connect_confirm", lang),
        parse_mode=ParseMode.HTML,
        reply_markup=connected_keyboard())


# ---------------- handlers ----------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if is_admin(user.id):
        await update.message.reply_text(
            WELCOME_ADMIN, parse_mode=ParseMode.HTML,
            reply_markup=admin_keyboard())
        return
    DATA["users"].setdefault(str(user.id), {
        "name": user.full_name,
        "username": user.username or "",
    })
    save_data(DATA)
    lang = get_lang(user.id)
    if user.id in CONNECTED:
        await update.message.reply_text(
            t("connected_start", lang),
            reply_markup=connected_keyboard())
    else:
        await update.message.reply_text(
            t("welcome", lang), parse_mode=ParseMode.HTML,
            reply_markup=main_keyboard())


async def send_stats(msg) -> None:
    await msg.reply_text(
        f"📊 <b>পরিসংখ্যান</b>\n\n"
        f"👥 ব্যবহারকারী: {len(DATA['users'])}\n"
        f"📨 পাঠানো মেসেজ: {DATA.get('forwarded', 0)}",
        parse_mode=ParseMode.HTML,
        reply_markup=admin_keyboard())


async def do_broadcast(context: ContextTypes.DEFAULT_TYPE, text: str,
                       msg) -> None:
    ok = fail = 0
    for uid_str in list(DATA.get("users", {})):
        uid = int(uid_str)
        if uid == ADMIN_ID:
            continue
        try:
            await context.bot.send_message(chat_id=uid, text=text)
            ok += 1
        except Exception as e:
            fail += 1
            log.warning("broadcast to %s failed: %s", uid, e)
    await msg.reply_text(
        f"📢 <b>ব্রডকাস্ট পাঠানো হয়েছে</b>\n\n"
        f"✅ {ok} জন পেয়েছে\n"
        f"❌ {fail} জন পায়নি",
        parse_mode=ParseMode.HTML,
        reply_markup=admin_keyboard())
    log.info("broadcast: ok=%s fail=%s", ok, fail)


async def handle_admin_message(update: Update,
                               context: ContextTypes.DEFAULT_TYPE,
                               msg, text: str, plain: str) -> None:
    # Text-to-Voice toggle button -> flip the state, save it, and
    # confirm with the updated keyboard. Checked before the
    # broadcast logic so a tap never starts a broadcast.
    if plain in (LBL_T2V_ON, LBL_T2V_OFF, LBL_T2V):
        enabled = not admin_voice_enabled()
        set_admin_voice(enabled)
        if enabled:
            confirm = ("🔊 টেক্সট-টু-ভয়েস চালু হয়েছে — এখন থেকে "
                       "আপনার উত্তর ইউজার ভয়েস হিসেবে পাবে।")
        else:
            confirm = ("🔇 টেক্সট-টু-ভয়েস বন্ধ হয়েছে — এখন থেকে "
                       "আপনার উত্তর ইউজার টেক্সট হিসেবে পাবে।")
        await msg.reply_text(confirm, reply_markup=admin_keyboard())
        log.info("admin text-to-voice toggled %s",
                 "on" if enabled else "off")
        return
    # Broadcast button -> next text message is the broadcast.
    if plain == LBL_BROADCAST:
        ADMIN_FLAGS["broadcast_pending"] = True
        await msg.reply_text(
            "📢 <b>ব্রডকাস্ট</b>\n\n"
            "ব্রডকাস্ট মেসেজটি লিখে পাঠান — সব ইউজারের কাছে যাবে।",
            parse_mode=ParseMode.HTML,
            reply_markup=admin_keyboard())
        return
    if plain == LBL_STATS:
        await send_stats(msg)
        return
    if ADMIN_FLAGS.get("broadcast_pending"):
        if text and not msg.reply_to_message:
            ADMIN_FLAGS["broadcast_pending"] = False
            await do_broadcast(context, text, msg)
            return
        ADMIN_FLAGS["broadcast_pending"] = False
        await msg.reply_text(
            "ব্রডকাস্টের জন্য লেখা মেসেজ পাঠান 🙂",
            reply_markup=admin_keyboard())
        return

    # Direct reply to a user's message -> deliver to that user.
    replied = msg.reply_to_message
    if replied is not None:
        target_id = DATA.get("threads", {}).get(str(replied.message_id))
        if target_id is None and replied.text:
            m = re.search(r"চ্যাট আইডি:\s*(\d+)", replied.text)
            if m:
                target_id = int(m.group(1))
        if target_id is None and getattr(replied, "forward_origin", None):
            sender = getattr(replied.forward_origin, "sender_user", None)
            if sender is not None:
                target_id = sender.id
        if target_id:
            try:
                if text:
                    # Text-to-Voice toggle decides: voice when ON,
                    # plain text when OFF.
                    await send_admin_reply(context, target_id, text)
                else:
                    await context.bot.copy_message(
                        chat_id=target_id,
                        from_chat_id=msg.chat_id,
                        message_id=msg.message_id)
                await msg.reply_text("✅ উত্তর পাঠানো হয়েছে।")
                record_admin_reply(text)
                log.info("admin replied to user %s", target_id)
            except Exception as e:
                await msg.reply_text(f"❌ পাঠানো যায়নি: {e}")
            return
    await msg.reply_text(
        "👑 উত্তর দিতে ইউজারের মেসেজে <b>রিপ্লাই</b> দিন,\n"
        "অথবা ব্যবহার করুন: <code>/reply user_id মেসেজ</code>",
        parse_mode=ParseMode.HTML,
        reply_markup=admin_keyboard())


async def handle_message(update: Update,
                         context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    user = update.effective_user
    if not msg or not user:
        return
    text = (msg.text or "").strip()
    # Button taps arrive in the bold font; normalize for matching.
    plain = from_bold(text)

    # --- Admin side ---
    if is_admin(user.id):
        await handle_admin_message(update, context, msg, text, plain)
        return

    # --- Spam auto-block (feature 11) ---
    if is_blocked(user.id):
        return  # silently ignore blocked users
    spam_reason = check_spam(user.id, text or (msg.caption or ""))
    if spam_reason:
        block_user(user.id, spam_reason)
        try:
            await context.bot.send_message(
                chat_id=ADMIN_ID,
                text=(f"🚫 <b>স্প্যাম: ইউজার ব্লক হয়েছে</b>\n\n"
                      f"নাম: {user.first_name}\n"
                      f"চ্যাট আইডি: {user.id}\n"
                      f"কারণ: {spam_reason}\n\n"
                      f"আনব্লক করতে: /unblock {user.id}"),
                parse_mode=ParseMode.HTML)
        except Exception as e:
            log.warning("spam notify failed: %s", e)
        await msg.reply_text(
            t("spam_blocked", get_lang(user.id)),
            reply_markup=main_keyboard())
        log.info("user %s auto-blocked: %s", user.id, spam_reason)
        return

    # --- Normal user ---

    # 1) Main-menu buttons.
    if plain == LBL_CONTACT:
        if user.id in CONNECTED:
            await msg.reply_text(
                t("already_connected", get_lang(user.id)),
                reply_markup=connected_keyboard())
            return
        # Feature 17: 00:00-07:00 the admin is offline; AI stays on.
        if admin_offline_now():
            await msg.reply_text(offline_msg(get_lang(user.id)),
                                 reply_markup=main_keyboard())
            return
        await connect_user(update, context, msg, user)
        return
    if plain == LBL_AI:
        lang = get_lang(user.id)
        if user.id in CONNECTED:
            await msg.reply_text(
                t("ai_connected_hint", lang),
                reply_markup=connected_keyboard())
            return
        await msg.reply_text(
            t("ai_on", lang),
            parse_mode=ParseMode.HTML, reply_markup=main_keyboard())
        return
    if plain == LBL_HELP:
        await msg.reply_text(help_text(get_lang(user.id)),
                             parse_mode=ParseMode.HTML,
                             reply_markup=main_keyboard())
        return
    if plain == LBL_OWNER:
        kb = (connected_keyboard() if user.id in CONNECTED
              else main_keyboard())
        await msg.reply_text(owner_text(get_lang(user.id)),
                             parse_mode=ParseMode.HTML, reply_markup=kb)
        add_history(user.id, "user", text)
        add_history(user.id, "bot", owner_plain())
        return
    if plain == LBL_LANG:
        # Feature 14 (corrected): show INLINE buttons; the user taps
        # one to select. No language change without an explicit tap.
        cur = get_lang(user.id)
        markup = InlineKeyboardMarkup([[
            InlineKeyboardButton(
                "🇧🇩 বাংলা" + (" ✅" if cur == "bn" else ""),
                callback_data="lang:bn", style="primary"),
            InlineKeyboardButton(
                "🇬🇧 English" + (" ✅" if cur == "en" else ""),
                callback_data="lang:en", style="primary"),
        ]])
        await msg.reply_text(
            "কোন ভাষায় উত্তর চান? 🙂\nWhich language do you want?",
            reply_markup=markup)
        return

    # 2) Live admin session: everything goes to the admin.
    if user.id in CONNECTED:
        if plain in (LBL_STOP, LBL_CANCEL) or (
                plain and plain.lower() in STOP_WORDS):
            CONNECTED.discard(user.id)
            await msg.reply_text(
                t("session_ended", get_lang(user.id)),
                parse_mode=ParseMode.HTML,
                reply_markup=main_keyboard())
            return
        if plain == LBL_AI:
            await msg.reply_text(
                t("ai_connected_hint", get_lang(user.id)),
                reply_markup=connected_keyboard())
            return
        if plain == LBL_HELP:
            await msg.reply_text(help_text(get_lang(user.id)),
                                 parse_mode=ParseMode.HTML,
                                 reply_markup=connected_keyboard())
            return
        await send_to_admin(update, context, msg)
        return

    # 3) Stop/Cancel typed while not connected.
    if plain and plain.lower() in STOP_WORDS:
        await msg.reply_text(
            t("not_connected", get_lang(user.id)),
            reply_markup=main_keyboard())
        return

    # 4) Owner question: answer from the intro, do NOT connect.
    if text and any(w in text.lower() for w in OWNER_WORDS):
        reply = owner_text(get_lang(user.id))
        await msg.reply_text(reply, parse_mode=ParseMode.HTML,
                             reply_markup=main_keyboard())
        add_history(user.id, "user", text)
        add_history(user.id, "bot", owner_plain())
        log.info("owner intro shown to user %s", user.id)
        return

    # 5) User explicitly wants the admin: connect and send this message.
    if text and any(w in text.lower() for w in ADMIN_WORDS):
        # Feature 17: 00:00-07:00 the admin is offline; AI stays on.
        if admin_offline_now():
            await msg.reply_text(offline_msg(get_lang(user.id)),
                                 reply_markup=main_keyboard())
            return
        await connect_user(update, context, msg, user)
        await send_to_admin(update, context, msg)
        return

    # 5b) Voice message: transcribe, then answer like a text message.
    if msg.voice:
        transcript = await transcribe_voice_message(context, msg)
        if not transcript:
            await msg.reply_text(
                t("voice_not_understood", get_lang(user.id)),
                reply_markup=main_keyboard())
            return
        if is_urgent(transcript):
            await urgent_alert(context, user, transcript)
        lang = get_lang(user.id)
        reply = ai_reply(transcript, get_history(user.id), lang)
        # AI replies are always text — never voice, even for a
        # voice-initiated chat (Abu, 2026-11-24).
        await send_ai_answer(msg, user, transcript, reply, lang)
        return

    # 5c) Photo: describe it with a vision model, then answer.
    if msg.photo:
        lang = get_lang(user.id)
        answer = await describe_photo_message(context, msg, lang)
        if not answer:
            await msg.reply_text(
                t("photo_not_understood", get_lang(user.id)),
                reply_markup=main_keyboard())
            return
        question = (msg.caption or "").strip() or (
            "photo" if lang == "en" else "ছবি")
        await send_ai_answer(msg, user, question, answer, lang)
        return

    # 6) Auto AI reply (with the last 10 messages as context).
    if text:
        if is_urgent(text):
            await urgent_alert(context, user, text)
        lang = get_lang(user.id)
        reply = ai_reply(text, get_history(user.id), lang)
        await send_ai_answer(msg, user, text, reply, lang)
        return

    # Non-text (photo/sticker/...) with no session.
    await msg.reply_text(
        t("media_fallback", get_lang(user.id)),
        reply_markup=main_keyboard())


async def reply_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_admin(user.id):
        return
    args = context.args
    if len(args) < 2:
        await update.message.reply_text("ব্যবহার: /reply user_id মেসেজ")
        return
    try:
        target_id = int(args[0])
    except ValueError:
        await update.message.reply_text("user_id সংখ্যায় লিখুন।")
        return
    text = " ".join(args[1:])
    try:
        # Text-to-Voice toggle decides: voice when ON (text
        # fallback if TTS fails), plain text when OFF.
        await send_admin_reply(context, target_id, text)
        await update.message.reply_text("✅ উত্তর পাঠানো হয়েছে।")
        record_admin_reply(text)
        log.info("admin /reply to user %s", target_id)
    except Exception as e:
        await update.message.reply_text(f"❌ পাঠানো যায়নি: {e}")


async def broadcast_cmd(update: Update,
                        context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_admin(user.id):
        return
    text = " ".join(context.args) if context.args else ""
    if not text:
        await update.message.reply_text(
            "ব্যবহার: /broadcast মেসেজ\n"
            f"অথবা {BTN_BROADCAST} বোতাম চাপ দিয়ে মেসেজ লিখুন।")
        return
    await do_broadcast(context, text, update.message)


async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_admin(user.id):
        return
    await send_stats(update.message)


async def owner_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    kb = admin_keyboard() if is_admin(user.id) else main_keyboard()
    _lang = "bn" if is_admin(user.id) else get_lang(user.id)
    await update.message.reply_text(owner_text(_lang),
                                    parse_mode=ParseMode.HTML,
                                    reply_markup=kb)


async def _admin_only(update: Update) -> bool:
    if not is_admin(update.effective_user.id):
        await update.message.reply_text(
            "এই কমান্ড শুধু এডমিন ব্যবহার করতে পারবে।")
        return False
    return True


async def setintro_cmd(update: Update,
                       context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _admin_only(update):
        return
    text = " ".join(context.args) if context.args else ""
    if not text:
        await update.message.reply_text(
            "ব্যবহার: /setintro আপনার পরিচয়\n\n"
            "উদাহরণ:\n"
            "/setintro আমি আবু তাহের — এই বটের মালিক ও এডমিন।")
        return
    INFO["intro"] = text
    save_info(INFO)
    await update.message.reply_text(
        "✅ পরিচয় সেভ হয়েছে।\n\n" + owner_text("bn"),
        parse_mode=ParseMode.HTML, reply_markup=admin_keyboard())
    log.info("admin set intro: %s", text[:80])


async def addinfo_cmd(update: Update,
                      context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _admin_only(update):
        return
    text = " ".join(context.args) if context.args else ""
    if not text:
        await update.message.reply_text(
            "ব্যবহার: /addinfo তথ্য\n\n"
            "উদাহরণ:\n"
            "/addinfo চ্যানেল: https://t.me/MrAbuTaher")
        return
    INFO.setdefault("facts", []).append(text)
    save_info(INFO)
    await update.message.reply_text(
        "✅ তথ্য যোগ হয়েছে।\n\n" + owner_text("bn"),
        parse_mode=ParseMode.HTML, reply_markup=admin_keyboard())


async def listinfo_cmd(update: Update,
                       context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _admin_only(update):
        return
    facts = INFO.get("facts", [])
    lines = ["🪪 <b>বটের তথ্য</b>\n\n" + html.escape(owner_plain())]
    if facts:
        lines.append("\n📌 তথ্যগুলো:")
        lines.extend(f"{i}. {html.escape(f)}"
                     for i, f in enumerate(facts, 1))
    await update.message.reply_text(
        "\n".join(lines), parse_mode=ParseMode.HTML,
        reply_markup=admin_keyboard())


async def delinfo_cmd(update: Update,
                      context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _admin_only(update):
        return
    args = context.args
    if not args:
        await update.message.reply_text(
            "ব্যবহার: /delinfo নম্বর\n"
            "নম্বর দেখতে: /listinfo")
        return
    try:
        idx = int(args[0]) - 1
    except ValueError:
        await update.message.reply_text("নম্বর সংখ্যায় লিখুন।")
        return
    facts = INFO.get("facts", [])
    if idx < 0 or idx >= len(facts):
        await update.message.reply_text("এই নম্বরে কোনো তথ্য নেই।")
        return
    removed = facts.pop(idx)
    save_info(INFO)
    await update.message.reply_text(
        f"🗑️ মুছে ফেলা হয়েছে: {removed}",
        reply_markup=admin_keyboard())


async def unblock_cmd(update: Update,
                      context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _admin_only(update):
        return
    args = context.args
    if not args:
        await update.message.reply_text("ব্যবহার: /unblock user_id")
        return
    try:
        target_id = int(args[0])
    except ValueError:
        await update.message.reply_text("user_id সংখ্যায় লিখুন।")
        return
    if str(target_id) not in BLOCKED:
        await update.message.reply_text("এই ইউজার ব্লক নয়।")
        return
    del BLOCKED[str(target_id)]
    save_blocked(BLOCKED)
    await update.message.reply_text(
        f"✅ ইউজার {target_id} আনব্লক হয়েছে।",
        reply_markup=admin_keyboard())
    log.info("admin unblocked user %s", target_id)


async def poll_cmd(update: Update,
                   context: ContextTypes.DEFAULT_TYPE) -> None:
    """Feature 16 (admin only): send a Yes/No poll to all users."""
    if not await _admin_only(update):
        return
    question = " ".join(context.args) if context.args else ""
    if not question:
        await update.message.reply_text(
            "ব্যবহার: /poll প্রশ্ন\n\n"
            "উদাহরণ:\n/poll নতুন ফিচারটা কেমন লেগেছে?")
        return
    poll = {"question": question, "options": POLL_OPTIONS,
            "poll_ids": {}, "answers": {},
            "created": int(time.time())}
    ok = fail = 0
    for uid_str in list(DATA.get("users", {})):
        uid = int(uid_str)
        if uid == ADMIN_ID:
            continue
        try:
            sent = await context.bot.send_poll(
                chat_id=uid, question=question[:300],
                options=POLL_OPTIONS, is_anonymous=False)
            poll["poll_ids"][str(uid)] = sent.poll.id
            ok += 1
        except Exception as e:
            fail += 1
            log.warning("poll to %s failed: %s", uid, e)
    save_poll(poll)
    await update.message.reply_text(
        f"📊 <b>পোল পাঠানো হয়েছে</b>\n\n"
        f"প্রশ্ন: {html.escape(question)}\n"
        f"✅ {ok} জন পেয়েছে\n❌ {fail} জন পায়নি\n\n"
        "ফলাফল দেখতে: /pollresults",
        parse_mode=ParseMode.HTML, reply_markup=admin_keyboard())
    log.info("poll sent: ok=%s fail=%s", ok, fail)


async def pollresults_cmd(update: Update,
                          context: ContextTypes.DEFAULT_TYPE) -> None:
    """Feature 16 (admin only): show the active poll's tally."""
    if not await _admin_only(update):
        return
    poll = load_poll()
    if not poll:
        await update.message.reply_text("এখনো কোনো পোল পাঠানো হয়নি।")
        return
    answers = poll.get("answers", {})
    options = poll.get("options", POLL_OPTIONS)
    counts = [0] * len(options)
    for opt_idx in answers.values():
        if 0 <= opt_idx < len(counts):
            counts[opt_idx] += 1
    total = sum(counts)
    lines = [f"📊 <b>পোলের ফলাফল</b>\n\n"
             f"প্রশ্ন: {html.escape(poll.get('question', ''))}\n"]
    for opt, c in zip(options, counts):
        pct = (c * 100 // total) if total else 0
        lines.append(f"{html.escape(opt)}: {c} ({pct}%)")
    lines.append(f"\nমোট উত্তর: {total} টি")
    await update.message.reply_text(
        "\n".join(lines), parse_mode=ParseMode.HTML,
        reply_markup=admin_keyboard())


async def on_poll_answer(update: Update,
                         context: ContextTypes.DEFAULT_TYPE) -> None:
    """Feature 16: record one user's poll answer."""
    ans = update.poll_answer
    if not ans:
        return
    poll = load_poll()
    if not poll or not ans.option_ids:
        return
    answers = poll.setdefault("answers", {})
    answers[str(ans.user.id)] = ans.option_ids[0]
    save_poll(poll)
    log.info("poll answer from user %s: option %s",
             ans.user.id, ans.option_ids[0])


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log the error and tell the user to retry instead of going silent."""
    log.error("handler error: %s", context.error)
    try:
        if isinstance(update, Update) and update.effective_message:
            _u = update.effective_user
            _lang = get_lang(_u.id) if _u else "bn"
            await update.effective_message.reply_text(
                t("error_retry", _lang))
    except Exception:
        pass


def build_app() -> Application:
    """Build a fresh Application with the same handlers and timeouts."""
    request = HTTPXRequest(
        connect_timeout=30.0, read_timeout=30.0,
        write_timeout=30.0, pool_timeout=30.0)
    app = Application.builder().token(TOKEN).request(request).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reply", reply_cmd))
    app.add_handler(CommandHandler("broadcast", broadcast_cmd))
    app.add_handler(CommandHandler("stats", stats_cmd))
    app.add_handler(CommandHandler("owner", owner_cmd))
    app.add_handler(CommandHandler("setintro", setintro_cmd))
    app.add_handler(CommandHandler("addinfo", addinfo_cmd))
    app.add_handler(CommandHandler("listinfo", listinfo_cmd))
    app.add_handler(CommandHandler("delinfo", delinfo_cmd))
    app.add_handler(CommandHandler("unblock", unblock_cmd))
    app.add_handler(CommandHandler("poll", poll_cmd))
    app.add_handler(CommandHandler("pollresults", pollresults_cmd))
    app.add_handler(PollAnswerHandler(on_poll_answer))
    app.add_handler(CallbackQueryHandler(rate_callback,
                                         pattern=r"^rate:(good|bad):"))
    app.add_handler(CallbackQueryHandler(lang_callback,
                                         pattern=r"^lang:(bn|en)$"))
    app.add_handler(CallbackQueryHandler(translate_callback,
                                         pattern=r"^tr:"))
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND,
                                    handle_message))
    app.add_error_handler(error_handler)
    return app


# Errors that mean "the network dropped": retry inside the process with
# backoff instead of letting the process die (the ~2h
# httpx.RemoteProtocolError crash loop — PTB wraps it as NetworkError,
# and the traceback used to vanish into /dev/null). Anything fatal at
# poll time exits instead, so the watchdog can restart a genuinely
# broken process.
TRANSIENT_ERRORS = (
    NetworkError,          # incl. TimedOut
    RetryAfter,
    httpx.TransportError,  # incl. RemoteProtocolError, ConnectError
    httpx.TimeoutException,
    OSError,
)


def main() -> None:
    if not TOKEN:
        raise SystemExit("BOT_TOKEN missing in .env")
    if not ADMIN_ID:
        raise SystemExit("ADMIN_ID missing in .env")
    # Single-instance lock: flock releases automatically when this
    # process dies, so a watchdog restart after a real death is never
    # blocked, but a second copy cannot start while this one runs.
    lock_file = open(BASE / "bot.lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.error("another instance is already running (bot.lock held); "
                  "exiting")
        raise SystemExit("another instance is already running")
    log.info("starting contact bot v6 (polling), admin=%s, gemini=%s, "
             "openrouter=%s, token=%s",
             ADMIN_ID, "on" if GEMINI_KEY else "off",
             "on" if OPENROUTER_AVAILABLE else "off", TOKEN_SOURCE)
    delay = 5.0
    while True:
        app = build_app()
        started = time.monotonic()
        try:
            # close_loop=False: run_polling closes the event loop by
            # default, and a second run_polling in this process would
            # then fail with "Event loop is closed". Keep it open so
            # the retry below can start a fresh Application.
            app.run_polling(allowed_updates=Update.ALL_TYPES,
                            close_loop=False)
            log.info("polling stopped cleanly; exiting")
            return
        except (InvalidToken, Forbidden, Conflict) as exc:
            # Bad/revoked token, or another instance polling with the
            # same token: retrying in-process cannot fix this. Exit;
            # the watchdog restarts a genuinely dead process.
            log.error("fatal polling error (%s): %s — exiting",
                      type(exc).__name__, exc)
            raise SystemExit(1)
        except TRANSIENT_ERRORS as exc:
            if isinstance(exc, RetryAfter):
                delay = max(delay, float(exc.retry_after))
            if time.monotonic() - started > 60:
                delay = 5.0  # healthy long run; reset the backoff
            log.exception("polling dropped (%s): %s — retrying in %.0fs",
                          type(exc).__name__, exc, delay)
            # run_polling installs loop signal handlers that only fire
            # while the loop runs; during this sleep SIGTERM would
            # otherwise be swallowed. Restore the defaults so a stop
            # request kills the process here.
            for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGABRT):
                signal.signal(_sig, signal.SIG_DFL)
            time.sleep(delay)
            delay = min(delay * 2, 300.0)


if __name__ == "__main__":
    main()
