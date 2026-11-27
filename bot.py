"""Admin contact bot v6 (inbox only, no mini app, no AI).

- ReplyKeyboardMarkup buttons only. Button labels use
  Mathematical Sans-Serif Bold Unicode (𝐀-𝐙, 𝐚-𝐳, 𝟎-𝟗);
  message bodies stay in normal Bengali. Every button carries
  an emoji prefix.
- Button colours via Bot API 9.4 `style` (primary = blue,
  success = green, danger = red). Inline buttons use `style`
  too: Join = primary, Verify = success, language picker and
  Translate buttons = primary.
- NO AI anywhere (removed 2026-11-27 on Abu's order): every
  user message — text, voice, photo, any media — goes straight
  to the admin as an inbox message, in this format:
      নাম: first_name
      ইউজারনেম: @username
      চ্যাট আইডি: user_id
      মেসেজ👇

      <message>
  The admin replies directly to that message and it is delivered
  to the user (admin-message-id -> user-id mapping, with a
  chat-ID regex fallback). /reply user_id text stays as backup.
- User messages are NEVER forwarded. Voice/photo/other media
  reach the admin as a copy (copy_message), never a forward.
- Admin broadcast: /broadcast text, or the Broadcast button
  followed by the message text.
- Owner identity: the bot knows its owner (intro + facts stored in
  data/bot_info.json). Admin commands: /setintro, /addinfo,
  /listinfo, /delinfo; anyone can see it with /owner or the Owner
  button. Owner questions ("এডমিন কে?") are answered from the
  intro instead of being sent on.
- Urgent words (জরুরি, টাকা ফেরত, সমস্যা, ...) send the admin an
  immediate separate alert. Spam (10+ msgs/60s or the same message
  3x) auto-blocks the user (data/blocked.json); /unblock lifts it.
- Nightly summary at 23:55 via daily_summary.py + the
  mr-support-nightly-summary cron (counts only, no AI).
- Language button (🌐 Language) shows inline buttons to pick
  Bengali/English/Chinese per user; the WHOLE bot follows it.
  Admin-facing strings stay Bengali.
- Each admin message carries two inline Translate buttons
  (🇧🇩 Bn / 🇬🇧 En, Groq first, Pollinations fallback); tapping one
  shows the translation below the original (tap again to hide).
  The admin's replies are auto-translated to the user's language
  when they differ. The admin has a "Text to Voice" toggle button
  (data/admin_voice.json): when ON, his typed replies reach the
  user as a voice message (Cartesia clone voice first, local tts
  fallback); when OFF (default), replies go as plain text.
- Media sharing ("𝐒𝐡𝐚𝐫𝐞 𝐥𝐢𝐧𝐤" admin button): the admin stores media
  in a private storage channel (data/storage_channel.json) and gets a
  deep link https://t.me/Mr_Chat1bot?start=00001 per file (sequential
  5-digit IDs in data/media.db). Anyone opening the link gets the file
  fresh from its file_id — never forwarded, so no "Forwarded from"
  tag — with protect_content=True so it cannot be forwarded or saved
  onward. Submenu: Link list / Remove link / Create link / Cancel.
- Force join (global, whole bot): using the bot requires being a
  member of the force-join group first (data/force_join.json).
  Non-members get a Join button + a "Verify" re-check button.
  The admin is exempt. The group defaults to the code constant
  FORCE_JOIN_DEFAULT_CHAT_ID (-1004367584470); a group set
  explicitly in the JSON (Force Create setup) overrides it.
  Leaving the group sends the user one private Bengali reminder.
  Force Create (admin button "🔐 𝐅𝐨𝐫𝐜𝐞 𝐂𝐫𝐞𝐚𝐭𝐞"): submenu with
  Force Join (set up by forwarding a group/channel message, or by
  @username / -100 ID — the bot must already be a member there),
  Remove (asks first, Yes/No confirmation) and List (show the
  current group). The Join button in the gate message shows the
  name given at setup (default "𝐌ʀ 𝐔ᴘᴅᴀᴛᴇꜱ", stored as
  "button_name").
"""

import asyncio
import fcntl
import html
import io
import json
import logging
import os
import re
import signal
import sqlite3
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
    ChatJoinRequestHandler,
    ChatMemberHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
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

# OpenRouter (custom.openrouter) serves the Groq model for
# translations (translate_text). The credential lives in the
# registered connection, never in a file: each call fetches an
# hsurr: surrogate from authd and lets it be swapped for the real
# key on the approved request to openrouter.ai.
# Model ID note (verified 2026-11-23): OpenRouter no longer lists a
# "groq/llama-3.3-70b-versatile" ID — the API answers 400 "not a valid
# model ID" and the /models list has zero groq/ IDs. Groq serves the
# same Llama 3.3 70B weights under the meta-llama ID; provider order
# ["groq"] pins the request to Groq (response field provider=Groq).
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct"
# Single-flight lock for OpenRouter calls (translate only now):
# the free tier answers HTTP 402 "in-flight budget exhausted"
# when several requests run at once, so at most one OpenRouter
# request is in flight at a time.
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

ADMIN_FLAGS = {"broadcast_pending": False}

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
        return body
    if lang == "zh":
        body = ("👑 <b>机器人主人 / 管理员</b>\n\n"
                + html.escape(INFO.get("intro", DEFAULT_INTRO)))
        facts = INFO.get("facts", [])
        if facts:
            body += "\n\n📌 <b>须知：</b>\n" + "\n".join(
                f"• {html.escape(f)}" for f in facts)
        return body
    body = ("👑 <b>বটের মালিক / এডমিন</b>\n\n"
            + html.escape(INFO.get("intro", DEFAULT_INTRO)))
    facts = INFO.get("facts", [])
    if facts:
        body += "\n\n📌 <b>জেনে রাখুন:</b>\n" + "\n".join(
            f"• {html.escape(f)}" for f in facts)
    return body


# ---------------- Language choice (feature 14) ----------------
# Per-user language: "bn" (default), "en" or "zh". The WHOLE bot
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
# no text-to-voice button.
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


# Dhaka timezone (used for media timestamps).
DHAKA = ZoneInfo("Asia/Dhaka")


# ---------------- Media sharing: "Share link" ----------------
# Admin-only. The admin stores a media file in a private storage
# channel and gets a deep link https://t.me/Mr_Chat1bot?start=00001.
# Retrieval sends the media fresh from its Telegram file_id — NEVER
# forwarded (no "Forwarded from" tag) — with protect_content=True so
# nobody can forward or save the shared file onward. The storage
# copy is also a copy_message (never forward_message).
# Storage channel: data/storage_channel.json {"channel_id": -100...}
# (survives restarts; the watchdog launches plain bot.py, so no env
# var is needed). IDs are sequential 5-digit numbers in data/media.db
# (MAX(id)+1, AUTOINCREMENT — never duplicated, survives restarts).
BOT_USERNAME = "Mr_Chat1bot"
STORAGE_CHANNEL_FILE = BASE / "data" / "storage_channel.json"


def load_storage_channel() -> int | None:
    if STORAGE_CHANNEL_FILE.exists():
        try:
            data = json.loads(
                STORAGE_CHANNEL_FILE.read_text(encoding="utf-8"))
            cid = data.get("channel_id")
            if isinstance(cid, int):
                return cid
        except Exception:
            pass
    env_val = ENV.get("STORAGE_CHANNEL_ID") or os.environ.get(
        "STORAGE_CHANNEL_ID")
    try:
        return int(env_val) if env_val else None
    except (TypeError, ValueError):
        return None


STORAGE_CHANNEL_ID = load_storage_channel()

# Share-link submenu state, persisted in data/share_link.json so a
# restart does not lose it: {"menu": bool, "pending": "create" |
# "remove" | None}. "menu" = admin is inside the Share-link submenu;
# "pending" = waiting for one media file ("create") or one File ID
# ("remove").
SHARE_LINK_FILE = BASE / "data" / "share_link.json"


def load_share_link_state() -> dict:
    if SHARE_LINK_FILE.exists():
        try:
            items = json.loads(
                SHARE_LINK_FILE.read_text(encoding="utf-8"))
            return items if isinstance(items, dict) else {}
        except Exception:
            pass
    return {}


def save_share_link_state(state: dict) -> None:
    SHARE_LINK_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


SHARE_LINK = load_share_link_state()


def share_menu_active() -> bool:
    return bool(SHARE_LINK.get("menu", False))


def share_pending() -> str | None:
    return SHARE_LINK.get("pending")


def set_share_state(menu: bool, pending: str | None = None) -> None:
    SHARE_LINK["menu"] = bool(menu)
    SHARE_LINK["pending"] = pending
    save_share_link_state(SHARE_LINK)


def clear_share_state() -> None:
    SHARE_LINK["menu"] = False
    SHARE_LINK["pending"] = None
    save_share_link_state(SHARE_LINK)


MEDIA_DB_FILE = BASE / "data" / "media.db"
MEDIA_LOCK = threading.Lock()


def _media_connect():
    conn = sqlite3.connect(MEDIA_DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def media_db_init() -> None:
    MEDIA_DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MEDIA_LOCK, _media_connect() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS media ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "file_id_5digit TEXT UNIQUE NOT NULL,"
            "storage_channel_id INTEGER NOT NULL,"
            "storage_message_id INTEGER NOT NULL,"
            "media_type TEXT NOT NULL,"
            "telegram_file_id TEXT NOT NULL,"
            "original_filename TEXT,"
            "original_caption TEXT,"
            "upload_date TEXT NOT NULL,"
            "uploader_id INTEGER NOT NULL)")
        conn.commit()


media_db_init()


def next_media_number() -> int:
    # sqlite_sequence (AUTOINCREMENT) keeps the high-water mark even
    # after the newest row is deleted, so a deleted File ID is never
    # reused for a different file.
    with MEDIA_LOCK, _media_connect() as conn:
        row = conn.execute(
            "SELECT seq FROM sqlite_sequence "
            "WHERE name = 'media'").fetchone()
        return int(row["seq"]) + 1 if row else 1


def media_insert(code: str, storage_channel_id: int,
                 storage_message_id: int, media_type: str,
                 telegram_file_id: str, original_filename: str | None,
                 original_caption: str, uploader_id: int) -> None:
    with MEDIA_LOCK, _media_connect() as conn:
        conn.execute(
            "INSERT INTO media (file_id_5digit, storage_channel_id,"
            " storage_message_id, media_type, telegram_file_id,"
            " original_filename, original_caption, upload_date,"
            " uploader_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (code, storage_channel_id, storage_message_id, media_type,
             telegram_file_id, original_filename, original_caption,
             datetime.now(DHAKA).strftime("%Y-%m-%d %H:%M:%S"),
             uploader_id))
        conn.commit()


def media_get_by_code(code: str):
    with MEDIA_LOCK, _media_connect() as conn:
        return conn.execute(
            "SELECT * FROM media WHERE file_id_5digit = ?",
            (code,)).fetchone()


def media_list_latest(limit: int = 15):
    with MEDIA_LOCK, _media_connect() as conn:
        return conn.execute(
            "SELECT * FROM media ORDER BY id DESC LIMIT ?",
            (limit,)).fetchall()


def media_stats() -> tuple:
    with MEDIA_LOCK, _media_connect() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM media").fetchone()[0]
        latest = conn.execute(
            "SELECT MAX(file_id_5digit) FROM media").fetchone()[0]
        by_type = dict(conn.execute(
            "SELECT media_type, COUNT(*) FROM media "
            "GROUP BY media_type").fetchall())
    return total, latest, by_type


def media_delete(code: str) -> None:
    with MEDIA_LOCK, _media_connect() as conn:
        conn.execute(
            "DELETE FROM media WHERE file_id_5digit = ?", (code,))
        conn.commit()


def media_share_link(code: str) -> str:
    return f"https://t.me/{BOT_USERNAME}?start={code}"


def media_storage_caption(code: str) -> str:
    return (f"File ID: {code}\n\nShare Link:\n{media_share_link(code)}")


def is_valid_media_code(code) -> bool:
    return bool(code) and re.fullmatch(r"\d{5}", str(code)) is not None


def extract_media(msg):
    """(media_type, telegram_file_id, original_filename) or None."""
    if msg.photo:
        return ("photo", msg.photo[-1].file_id, None)
    if msg.video:
        return ("video", msg.video.file_id,
                getattr(msg.video, "file_name", None))
    if msg.document:
        return ("document", msg.document.file_id,
                getattr(msg.document, "file_name", None))
    if msg.audio:
        return ("audio", msg.audio.file_id,
                getattr(msg.audio, "file_name", None))
    if msg.voice:
        return ("voice", msg.voice.file_id, None)
    if msg.animation:
        return ("animation", msg.animation.file_id,
                getattr(msg.animation, "file_name", None))
    if msg.sticker:
        return ("sticker", msg.sticker.file_id, None)
    if msg.video_note:
        return ("video_note", msg.video_note.file_id, None)
    return None


async def copy_media_to_storage(context, msg, caption: str) -> int:
    """Copy the admin's media message into the storage channel.
    copy_message only — never forward_message, so the stored copy
    carries no "Forwarded from" tag. protect_content=True blocks
    forwarding/saving of the stored copy. Returns the channel
    message id."""
    try:
        sent = await context.bot.copy_message(
            chat_id=STORAGE_CHANNEL_ID,
            from_chat_id=msg.chat_id,
            message_id=msg.message_id,
            caption=caption,
            protect_content=True)
        return sent.message_id
    except Exception:
        # Media that cannot carry a caption (video_note, sticker):
        # copy without one, then post the File ID + link as its own
        # message in the channel.
        log.info("copy with caption failed, retrying without caption",
                 exc_info=True)
        sent = await context.bot.copy_message(
            chat_id=STORAGE_CHANNEL_ID,
            from_chat_id=msg.chat_id,
            message_id=msg.message_id,
            protect_content=True)
        try:
            await context.bot.send_message(
                chat_id=STORAGE_CHANNEL_ID, text=caption,
                reply_to_message_id=sent.message_id,
                protect_content=True)
        except Exception:
            log.warning("storage info text failed", exc_info=True)
        return sent.message_id


async def send_stored_media(context, chat_id: int, row) -> bool:
    """Send stored media fresh from its Telegram file_id — never
    forwarded, so no "Forwarded from" tag. protect_content=True on
    every send blocks forwarding/saving. The ORIGINAL caption from
    the DB is used (never the channel's internal caption)."""
    mtype = row["media_type"]
    file_id = row["telegram_file_id"]
    caption = row["original_caption"] or None
    bot = context.bot
    try:
        if mtype == "photo":
            await bot.send_photo(chat_id, photo=file_id, caption=caption,
                                 protect_content=True)
        elif mtype == "video":
            await bot.send_video(chat_id, video=file_id, caption=caption,
                                 protect_content=True)
        elif mtype == "document":
            await bot.send_document(chat_id, document=file_id,
                                    caption=caption, protect_content=True)
        elif mtype == "audio":
            await bot.send_audio(chat_id, audio=file_id, caption=caption,
                                 protect_content=True)
        elif mtype == "voice":
            await bot.send_voice(chat_id, voice=file_id, caption=caption,
                                 protect_content=True)
        elif mtype == "animation":
            await bot.send_animation(chat_id, animation=file_id,
                                     caption=caption, protect_content=True)
        elif mtype == "sticker":
            await bot.send_sticker(chat_id, sticker=file_id,
                                   protect_content=True)
        elif mtype == "video_note":
            await bot.send_video_note(chat_id, video_note=file_id,
                                      protect_content=True)
        else:
            log.warning("send_stored_media: unknown type %s", mtype)
            return False
        if caption and mtype in ("sticker", "video_note"):
            await bot.send_message(chat_id, text=caption,
                                   protect_content=True)
        return True
    except Exception:
        log.exception("send_stored_media failed type=%s", mtype)
        return False


async def delete_media_by_code(context, code: str) -> str:
    """Delete the channel message AND the DB row.
    Returns 'ok' | 'not_found' | 'db_error'."""
    row = media_get_by_code(code)
    if row is None:
        return "not_found"
    try:
        await context.bot.delete_message(
            chat_id=int(row["storage_channel_id"]),
            message_id=int(row["storage_message_id"]))
    except Exception as e:
        # Message already gone from the channel: still drop the row.
        log.warning("mediadelete: channel message %s delete failed: %s",
                    row["storage_message_id"], e)
    try:
        media_delete(code)
    except Exception:
        log.exception("mediadelete: db delete failed for %s", code)
        return "db_error"
    return "ok"


async def handle_create_link_upload(update, context, msg, media) -> None:
    """One media file -> storage channel copy + DB row + share link."""
    user = update.effective_user
    if STORAGE_CHANNEL_ID is None:
        await msg.reply_text(
            "⚠️ স্টোরেজ চ্যানেল এখনো সেট করা হয়নি।\n"
            "চ্যানেল আইডি সেট করার পর আবার চেষ্টা করুন।",
            reply_markup=share_link_keyboard())
        log.warning("create link: storage channel not configured")
        return
    mtype, file_id, filename = media
    caption = (msg.caption or "").strip()
    try:
        code = f"{next_media_number():05d}"
        storage_msg_id = await copy_media_to_storage(
            context, msg, media_storage_caption(code))
        media_insert(code=code, storage_channel_id=STORAGE_CHANNEL_ID,
                     storage_message_id=storage_msg_id, media_type=mtype,
                     telegram_file_id=file_id, original_filename=filename,
                     original_caption=caption, uploader_id=user.id)
    except Exception:
        log.exception("create link upload failed")
        await msg.reply_text(
            "❌ লিংক তৈরি হয়নি, আবার চেষ্টা করুন।",
            reply_markup=share_link_keyboard())
        return
    await msg.reply_text(
        f"✅ <b>লিংক তৈরি হয়েছে</b>\n\n"
        f"File ID: <code>{code}</code>\n\n"
        f"Share Link:\n{media_share_link(code)}\n\n"
        f"আরেকটি লিংক বানাতে {BTN_CREATE_LINK} চাপুন।",
        parse_mode=ParseMode.HTML,
        reply_markup=share_link_keyboard())
    log.info("create link: %s stored as %s", mtype, code)


async def show_link_list(msg) -> None:
    """Latest 15 stored links, newest first (submenu)."""
    try:
        rows = media_list_latest(15)
    except Exception:
        log.exception("link list failed")
        await msg.reply_text(
            "❌ লিংক তালিকা দেখা যায়নি, আবার চেষ্টা করুন।",
            reply_markup=share_link_keyboard())
        return
    if not rows:
        await msg.reply_text(
            "📭 এখনো কোনো লিংক তৈরি হয়নি।\n"
            "নতুন লিংক বানাতে 𝐂𝐫𝐞𝐚𝐭𝐞 𝐥𝐢𝐧𝐤 চাপুন।",
            reply_markup=share_link_keyboard())
        return
    lines = ["🔗 <b>লিংক তালিকা</b> (সর্বশেষ 15টি)\n"]
    for row in rows:
        code = row["file_id_5digit"]
        lines.append(f"File ID: <code>{code}</code>\n"
                     f"{media_share_link(code)}")
    await msg.reply_text("\n\n".join(lines), parse_mode=ParseMode.HTML,
                         reply_markup=share_link_keyboard())


# ---------------- Force join (global, whole bot) ----------------
# Using the bot requires the user to be a member of the
# force-join group first. The admin is exempt. Config lives in
# data/force_join.json and is re-read on every check, so a newly
# recorded group takes effect without a restart. A group set
# explicitly in the JSON (Force Create setup) always wins over
# the code default below. The bot joining the default group only
# fills in its title (my_chat_member handler below).
FORCE_JOIN_FILE = BASE / "data" / "force_join.json"
FORCE_JOIN_INVITE = "https://t.me/+2EZrWZKxy2dmMzll"
# Default force-join group (Abu, 2026-11-27): the gate is active
# from the next restart even before the bot is added to the group.
# A group set explicitly in data/force_join.json (via the Force
# Join setup flow) always wins over this default.
FORCE_JOIN_DEFAULT_CHAT_ID = -1004367584470
# Default text of the Join button in the force-join gate message
# (Abu, 2026-11-27): "𝐌ʀ 𝐔ᴘᴅᴀᴛᴇꜱ". A custom name given at setup
# overwrites it in data/force_join.json ("button_name"). Written
# with escapes so the exact code points survive editing.
FORCE_JOIN_DEFAULT_BUTTON = (
    "\U0001D40C\u0280 \U0001D414\u1D18\u1D05"
    "\u1D00\u1D1B\u1D07\uA731"
)

# Statuses that count as "joined" for getChatMember.
FJ_ALLOWED_STATUSES = {"member", "administrator", "creator",
                       "restricted"}

FJ_PROMPT_TEXT = (
    "📢 <b>বট ব্যবহার করতে গ্রুপে জয়েন করুন</b>\n\n"
    "এই বটটি ব্যবহার করতে হলে প্রথমে নিচের গ্রুপে "
    "জয়েন করতে হবে।\n"
    "জয়েন করার পর \"Verify\" চাপুন।"
)


def load_force_join() -> dict:
    """Effective force-join config. A JSON-set chat_id always
    wins; when the JSON has none, the code default fills in and
    "from_default" marks the source (never persisted)."""
    if FORCE_JOIN_FILE.exists():
        try:
            data = json.loads(
                FORCE_JOIN_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("chat_id", None)
                data.setdefault("title", None)
                data.setdefault("invite_link", FORCE_JOIN_INVITE)
                data.setdefault("alerted", False)
                data.setdefault("menu", False)
                data.setdefault("setup_wait", False)
                data.setdefault("button_name",
                              FORCE_JOIN_DEFAULT_BUTTON)
                data.setdefault("pending", None)
                data.setdefault("bot_invite_link", None)
                data.setdefault("bot_invite_link_chat_id", None)
                if not data.get("chat_id"):
                    data["chat_id"] = FORCE_JOIN_DEFAULT_CHAT_ID
                    data["from_default"] = True
                else:
                    data["from_default"] = False
                return data
        except Exception:
            pass
    return {"chat_id": FORCE_JOIN_DEFAULT_CHAT_ID, "title": None,
            "invite_link": FORCE_JOIN_INVITE, "alerted": False,
            "menu": False, "setup_wait": False,
            "button_name": FORCE_JOIN_DEFAULT_BUTTON,
            "pending": None, "bot_invite_link": None,
            "bot_invite_link_chat_id": None, "from_default": True}


def save_force_join(data: dict) -> None:
    data = dict(data)
    if data.pop("from_default", False):
        # The chat_id came from the code default: do not persist
        # it as a JSON-set group.
        if data.get("chat_id") == FORCE_JOIN_DEFAULT_CHAT_ID:
            data["chat_id"] = None
    FORCE_JOIN_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8")


# Create the config file at startup if it does not exist yet.
if not FORCE_JOIN_FILE.exists():
    save_force_join(load_force_join())


async def ensure_bot_invite_link(bot) -> str | None:
    """The bot's own invite link for the force-join group (Abu,
    2026-11-27: the bot makes its own link, and only requests
    through that link are auto-approved). Created once per group
    with creates_join_request=True and stored in
    data/force_join.json; reused afterwards, never recreated on
    every start. Returns the URL or None (creation failed)."""
    cfg = load_force_join()
    chat_id = cfg.get("chat_id")
    if not chat_id:
        return None
    if (cfg.get("bot_invite_link")
            and cfg.get("bot_invite_link_chat_id") == chat_id):
        return cfg["bot_invite_link"]
    try:
        link = await bot.create_chat_invite_link(
            chat_id=chat_id, name="Mr Chat Bot",
            creates_join_request=True)
        url = getattr(link, "invite_link", None)
        if not url:
            return None
    except Exception as e:
        log.warning("force join: could not create own invite link "
                    "for %s: %s", chat_id, e)
        return None
    cfg["bot_invite_link"] = url
    cfg["bot_invite_link_chat_id"] = chat_id
    try:
        save_force_join(cfg)
    except Exception:
        log.exception("force join: could not save own invite link")
        return None
    log.info("force join: own invite link created for %s", chat_id)
    return url


async def force_join_post_init(app: Application) -> None:
    """On startup, make sure the bot's own force-join invite
    link exists (reused if already stored)."""
    try:
        await ensure_bot_invite_link(app.bot)
    except Exception:
        pass


def force_join_markup(code: str) -> InlineKeyboardMarkup:
    cfg = load_force_join()
    btn_text = cfg.get("button_name") or FORCE_JOIN_DEFAULT_BUTTON
    # The bot's own invite link first (Abu, 2026-11-27) — only
    # requests through it are auto-approved.
    join_url = (cfg.get("bot_invite_link") or cfg.get("invite_link")
                or FORCE_JOIN_INVITE)
    return InlineKeyboardMarkup([[
        InlineKeyboardButton(
            btn_text, url=join_url,
            style="primary"),
    ], [
        InlineKeyboardButton(
            "Verify", callback_data=f"fjcheck:{code}", style="success"),
    ]])


async def force_join_check(context, user_id: int) -> str:
    """'off' (no group configured) | 'allow' | 'block'.
    A getChatMember error fails OPEN (the user is still let
    through) and alerts the admin once."""
    cfg = load_force_join()
    chat_id = cfg.get("chat_id")
    if not chat_id:
        return "off"
    try:
        member = await context.bot.get_chat_member(
            chat_id=chat_id, user_id=user_id)
        if getattr(member, "status", "") in FJ_ALLOWED_STATUSES:
            if cfg.get("alerted"):
                cfg["alerted"] = False
                try:
                    save_force_join(cfg)
                except Exception:
                    pass
            return "allow"
        return "block"
    except Exception as e:
        log.warning("force join check failed for %s: %s", user_id, e)
        if not cfg.get("alerted"):
            cfg["alerted"] = True
            try:
                save_force_join(cfg)
            except Exception:
                pass
            try:
                await context.bot.send_message(
                    chat_id=ADMIN_ID,
                    text=(f"⚠️ <b>ফোর্স জয়েন পরীক্ষা ব্যর্থ</b>\n\n"
                          f"গ্রুপ: {cfg.get('title') or chat_id}\n"
                          f"কারণ: {e}\n"
                          f"ব্যবহারকারীকে তবুও ঢুকতে দেওয়া হচ্ছে।"),
                    parse_mode=ParseMode.HTML)
            except Exception:
                log.warning("force join admin alert failed",
                            exc_info=True)
        return "allow"


async def force_join_callback(update: Update,
                              context: ContextTypes.DEFAULT_TYPE) -> None:
    """'Verify' tap: re-check membership, then deliver."""
    query = update.callback_query
    if not query or not query.data:
        return
    try:
        _, code = query.data.split(":", 1)
    except ValueError:
        await query.answer()
        return
    if code == "start":
        # Plain /start (or any non-media message) context: after a
        # successful verify the user gets the normal welcome.
        user = update.effective_user
        gate = await force_join_check(context, user.id)
        if gate == "block":
            await query.answer("আপনি এখনো গ্রুপে জয়েন করেননি।")
            try:
                await query.edit_message_text(
                    FJ_PROMPT_TEXT, parse_mode=ParseMode.HTML,
                    reply_markup=force_join_markup("start"))
            except Exception:
                log.warning("force join re-prompt edit failed",
                            exc_info=True)
            return
        await query.answer("✅ ধন্যবাদ! স্বাগতম।")
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass
        chat_id = query.message.chat_id if query.message else user.id
        await send_user_welcome(context, chat_id, user)
        return
    if not is_valid_media_code(code):
        await query.answer("Sorry, this file could not be found.")
        return
    try:
        row = media_get_by_code(code)
    except Exception:
        log.exception("force join re-check lookup failed")
        row = None
    if row is None:
        await query.answer("Sorry, this file could not be found.")
        return
    user = update.effective_user
    gate = await force_join_check(context, user.id)
    if gate == "block":
        await query.answer("আপনি এখনো গ্রুপে জয়েন করেননি।")
        try:
            await query.edit_message_text(
                FJ_PROMPT_TEXT, parse_mode=ParseMode.HTML,
                reply_markup=force_join_markup(code))
        except Exception:
            log.warning("force join re-prompt edit failed",
                        exc_info=True)
        return
    await query.answer("✅ ধন্যবাদ! ফাইল পাঠানো হচ্ছে।")
    try:
        await query.edit_message_reply_markup(reply_markup=None)
    except Exception:
        pass
    chat_id = query.message.chat_id if query.message else user.id
    ok = await send_stored_media(context, chat_id, row)
    if not ok:
        await context.bot.send_message(
            chat_id=chat_id,
            text="Sorry, this file could not be found.")


async def force_remove_callback(update: Update,
                                 context: ContextTypes.DEFAULT_TYPE) -> None:
    """Yes/No answer to the Force Create Remove confirmation.

    Yes on a JSON-set group: clear chat_id + title from the JSON —
    the code default then takes over (the admin is told so). Yes
    on the code default itself: it lives in the code and cannot be
    removed from the bot. No: cancel, back to the submenu."""
    query = update.callback_query
    if not query or not query.data:
        return
    if not query.from_user or not is_admin(query.from_user.id):
        await query.answer()
        return
    try:
        _, answer = query.data.split(":", 1)
    except ValueError:
        await query.answer()
        return
    if answer != "yes":
        await query.answer("বাতিল করা হয়েছে।")
        try:
            await query.edit_message_text("❌ বাতিল করা হয়েছে।")
        except Exception:
            pass
        return
    cfg = load_force_join()
    if not cfg.get("chat_id"):
        await query.answer("কোনো ফোর্স জয়েন সেট করা নেই।")
        try:
            await query.edit_message_text("কোনো ফোর্স জয়েন সেট করা নেই।")
        except Exception:
            pass
        return
    if cfg.get("from_default"):
        await query.answer()
        try:
            await query.edit_message_text(
                "⚠️ <b>ডিফল্ট ফোর্স জয়েন মোছা যাবে না</b>\n\n"
                f"গ্রুপ: {cfg.get('title') or '—'}\n"
                f"চ্যাট আইডি: <code>{cfg['chat_id']}</code>\n\n"
                "এই গ্রুপটি বটের কোডে সেট করা আছে — বট থেকে "
                "মোছা যাবে না। বদলাতে হলে কোড বদলাতে হবে।",
                parse_mode=ParseMode.HTML)
        except Exception:
            pass
        log.info("force join: remove refused — group is the code "
                 "default")
        return
    title = cfg.get("title") or cfg["chat_id"]
    cfg["chat_id"] = None
    cfg["title"] = None
    cfg["alerted"] = False
    try:
        save_force_join(cfg)
    except Exception:
        log.exception("force join: remove save failed")
        await query.answer("❌ মোছা যায়নি।")
        return
    await query.answer("🗑️ ফোর্স জয়েন মুছে ফেলা হয়েছে।")
    try:
        await query.edit_message_text(
            "🗑️ <b>ফোর্স জয়েন মুছে ফেলা হয়েছে</b>\n\n"
            f"মোছা হয়েছে: {html.escape(str(title))}\n\n"
            "এখন থেকে ডিফল্ট গ্রুপটি কার্যকর হবে:\n"
            f"চ্যাট আইডি: <code>{FORCE_JOIN_DEFAULT_CHAT_ID}</code>",
            parse_mode=ParseMode.HTML)
    except Exception:
        pass
    log.info("force join: JSON-set group removed by admin; code "
             "default %s now in effect", FORCE_JOIN_DEFAULT_CHAT_ID)


async def my_chat_member_update(update: Update,
                                context: ContextTypes.DEFAULT_TYPE) -> None:
    """The bot itself was added to / removed from a group or channel.

    With a code default group present, auto-capture no longer
    records the first group the bot joins: if the added chat IS
    the code-default group, only its title is recorded into the
    JSON (the chat_id itself stays unset there — the code default
    provides it); any other group is just logged, never recorded.
    A group set explicitly in the JSON is never overwritten."""
    cm = update.my_chat_member
    if not cm:
        return
    new_status = getattr(cm.new_chat_member, "status", "")
    old_status = getattr(cm.old_chat_member, "status", "")
    if (new_status in ("member", "administrator")
            and old_status in ("left", "kicked")):
        chat = cm.chat
        cfg = load_force_join()
        if not cfg.get("from_default"):
            # A group is explicitly set in the JSON: never
            # overwrite it with an auto-capture.
            log.info("force join: bot added to %s (%s), already "
                     "configured for %s",
                     chat.title, chat.id, cfg.get("chat_id"))
            return
        if chat.id != FORCE_JOIN_DEFAULT_CHAT_ID:
            # Not the default group: do not record it — the code
            # default stays in effect.
            log.info("force join: bot added to %s (%s), not the "
                     "default group — not recorded",
                     chat.title, chat.id)
            return
        # The bot joined the code-default group: record its title
        # (the chat_id itself stays unset in the JSON; the code
        # default provides it).
        if cfg.get("title"):
            log.info("force join: bot added to default group %s, "
                     "title already known", chat.id)
            return
        cfg["title"] = chat.title
        try:
            save_force_join(cfg)
        except Exception:
            log.exception("force join: could not save title")
            return
        log.info("force join: recorded title %s for default group %s",
                 chat.title, chat.id)
        try:
            await context.bot.send_message(
                chat_id=ADMIN_ID,
                text=(f"✅ <b>ফোর্স জয়েন গ্রুপের নাম রেকর্ড হয়েছে</b>\n\n"
                      f"গ্রুপ: {chat.title}\n"
                      f"চ্যাট আইডি: <code>{chat.id}</code>\n\n"
                      f"বট ব্যবহার করতে ইউজারদের এই গ্রুপে "
                      f"জয়েন করতে হবে।"),
                parse_mode=ParseMode.HTML)
        except Exception:
            log.warning("force join notify failed", exc_info=True)


FJ_LEAVE_TEXT = (
    "👋 <b>আপনি গ্রুপ থেকে লিভ করেছেন</b>\n\n"
    "আপনি ফোর্স জয়েন গ্রুপ থেকে লিভ করেছেন। এই বটটি "
    "ব্যবহার করতে হলে আবার গ্রুপে জয়েন করতে হবে।\n\n"
    "নিচের বাটনে চাপ দিয়ে জয়েন করুন, তারপর "
    "\"Verify\" চাপুন।"
)


async def force_join_member_update(update: Update,
                                   context: ContextTypes.DEFAULT_TYPE) -> None:
    """A user left / was kicked from the force-join chat: send one
    private reminder with the Join button. These chat_member
    updates only arrive while the bot is an administrator in
    that chat. Never crashes: a user who never started the bot
    (or blocked it) simply cannot be messaged."""
    cm = update.chat_member
    if not cm:
        return
    cfg = load_force_join()
    chat_id = cfg.get("chat_id")
    if not chat_id or cm.chat.id != chat_id:
        return
    new_status = getattr(cm.new_chat_member, "status", "")
    old_status = getattr(cm.old_chat_member, "status", "")
    if new_status not in ("left", "kicked"):
        return
    if old_status in ("left", "kicked"):
        return  # was not a member before this event
    user = getattr(cm.new_chat_member, "user", None)
    if user is None or getattr(user, "is_bot", False):
        return
    if user.id == ADMIN_ID:
        return
    try:
        await context.bot.send_message(
            chat_id=user.id, text=FJ_LEAVE_TEXT,
            parse_mode=ParseMode.HTML,
            reply_markup=force_join_markup("start"))
        log.info("force join: leave reminder sent to %s", user.id)
    except Exception as e:
        log.info("force join: leave reminder to %s failed: %s",
                 user.id, e)


async def force_join_request_update(update: Update,
                                    context: ContextTypes.DEFAULT_TYPE) -> None:
    """A user asked to join the force-join chat: approve them
    automatically (Abu, 2026-11-27) — but ONLY if the request
    came through the bot's own invite link. Needs the bot to be
    an administrator there with can_invite_users; a failure is
    only logged, never a crash. Join requests only exist when
    the group requires approval for new members — otherwise
    people join directly and there is nothing to approve."""
    req = update.chat_join_request
    if not req:
        return
    cfg = load_force_join()
    chat_id = cfg.get("chat_id")
    if not chat_id or req.chat.id != chat_id:
        return
    user = getattr(req, "from_user", None)
    if user is None or getattr(user, "is_bot", False):
        return
    # Abu (2026-11-27): approve ONLY requests that came through
    # the bot's own invite link.
    bot_link = cfg.get("bot_invite_link")
    req_link = getattr(req, "invite_link", None)
    req_url = getattr(req_link, "invite_link", None) if req_link else None
    if not bot_link or req_url != bot_link:
        return
    try:
        await context.bot.approve_chat_join_request(
            chat_id=chat_id, user_id=user.id)
        log.info("force join: auto-approved join request from %s",
                 user.id)
    except Exception as e:
        log.info("force join: auto-approve failed for %s: %s",
                 user.id, e)


# ---------------- Force Create submenu state ----------------
# Stored in the same data/force_join.json (re-read on every
# access, so a change takes effect without a restart):
# "menu" = admin is inside the Force Create submenu;
# "setup_wait" = waiting for a forwarded message / @username / ID
# that identifies the force-join group.
def force_menu_active() -> bool:
    return bool(load_force_join().get("menu", False))


def force_setup_wait() -> bool:
    return bool(load_force_join().get("setup_wait", False))


def force_pending() -> str | None:
    return load_force_join().get("pending")


def set_force_state(menu: bool, setup_wait: bool = False,
                    pending: str | None = None) -> None:
    cfg = load_force_join()
    cfg["menu"] = bool(menu)
    cfg["setup_wait"] = bool(setup_wait)
    cfg["pending"] = pending
    save_force_join(cfg)


def clear_force_state() -> None:
    cfg = load_force_join()
    cfg["menu"] = False
    cfg["setup_wait"] = False
    cfg["pending"] = None
    save_force_join(cfg)


async def _force_setup_save(context, chat_id: int,
                            title: str | None) -> bool:
    """Verify the bot is a member of chat_id, then save it as the
    force-join group. Returns True on success."""
    bot_id = getattr(context.bot, "id", None)
    if not bot_id:
        try:
            bot_id = (await context.bot.get_me()).id
        except Exception:
            bot_id = None
    if not bot_id:
        return False
    try:
        member = await context.bot.get_chat_member(
            chat_id=chat_id, user_id=bot_id)
    except Exception as e:
        log.warning("force setup: bot membership check failed for "
                    "%s: %s", chat_id, e)
        return False
    if getattr(member, "status", "") not in ("member", "administrator",
                                             "creator"):
        return False
    cfg = load_force_join()
    cfg["chat_id"] = chat_id
    cfg["title"] = title
    cfg["from_default"] = False
    cfg["alerted"] = False
    cfg["menu"] = True
    cfg["setup_wait"] = False
    cfg["pending"] = "button_name"
    save_force_join(cfg)
    # The bot makes its own invite link for the new group (best
    # effort — a failure only logs; the Join button falls back
    # to the configured invite link).
    try:
        await ensure_bot_invite_link(context.bot)
    except Exception:
        pass
    return True


async def handle_force_setup_input(update, context, msg,
                                   text: str) -> bool:
    """Consume one admin message while Force Join setup is waiting.
    Returns True if the message was handled (setup done, or
    re-prompted), False if it should fall through to normal admin
    handling (a reply to a user)."""
    # A reply to a user is not setup input: let normal handling
    # take it.
    if msg.reply_to_message is not None:
        return False
    origin = getattr(msg, "forward_origin", None)
    origin_chat = None
    if origin is not None:
        origin_chat = (getattr(origin, "chat", None)
                       or getattr(origin, "sender_chat", None))
    target = None  # (chat_id, title) once resolved
    if (origin_chat is not None
            and getattr(origin_chat, "type", "") in
            ("group", "supergroup", "channel")):
        target = (origin_chat.id, origin_chat.title)
    else:
        ident = (text or "").strip()
        m = re.match(r"^(?:https?://)?t\.me/([A-Za-z0-9_]{5,})$",
                     ident)
        if m:
            ident = "@" + m.group(1)
        if re.fullmatch(r"-?\d+", ident or ""):
            try:
                chat = await context.bot.get_chat(chat_id=int(ident))
                target = (chat.id, chat.title)
            except Exception as e:
                log.warning("force setup: get_chat(%s) failed: %s",
                            ident, e)
        elif re.fullmatch(r"@[A-Za-z0-9_]{5,}", ident or ""):
            try:
                chat = await context.bot.get_chat(chat_id=ident)
                target = (chat.id, chat.title)
            except Exception as e:
                log.warning("force setup: get_chat(%s) failed: %s",
                            ident, e)
    if target is None:
        # Not a usable input: stay in setup wait and re-prompt.
        await msg.reply_text(
            "📌 <b>ফোর্স জয়েন সেটআপ</b>\n\n"
            "গ্রুপ/চ্যানেলের একটি মেসেজ ফরওয়ার্ড করুন, "
            "অথবা @username বা -100 ID পাঠান।",
            parse_mode=ParseMode.HTML,
            reply_markup=force_create_keyboard())
        return True
    chat_id, title = target
    if await _force_setup_save(context, chat_id, title):
        cfg = load_force_join()
        current = cfg.get("button_name") or FORCE_JOIN_DEFAULT_BUTTON
        await msg.reply_text(
            "✅ <b>ফোর্স জয়েন সেট হয়েছে</b>\n\n"
            f"গ্রুপ: {title}\n"
            f"চ্যাট আইডি: <code>{chat_id}</code>\n\n"
            "এখন থেকে শেয়ার লিংকের মিডিয়া পেতে "
            "ইউজারদের এই গ্রুপে জয়েন করতে হবে।\n\n"
            "জয়েন বাটনে কী নাম দেখাবে? নাম লিখে পাঠান।\n"
            f"(এখন: {html.escape(current)} — Cancel চাপলে এটাই থাকবে।)",
            parse_mode=ParseMode.HTML,
            reply_markup=force_create_keyboard())
        log.info("force join: set up %s (%s) by admin", title, chat_id)
        return True
    await msg.reply_text(
        "⚠️ <b>বট ওই গ্রুপে নেই</b>\n\n"
        "প্রথমে @Mr_Chat1bot কে ওই গ্রুপে এড করুন, "
        "তারপর আবার মেসেজটি ফরওয়ার্ড করুন বা ID পাঠান।",
        parse_mode=ParseMode.HTML,
        reply_markup=force_create_keyboard())
    return True


# ---------------- Spam auto-block (feature 11) ----------------
# 10+ messages in 60s, or the same message 3x in 60s, gets the user
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
    if len(times) >= SPAM_MAX_MSGS:
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


def strip_button_emoji(text: str) -> str:
    """Drop a leading emoji prefix (e.g. '🔗 ') from a button tap so
    '🔗 Share link' still matches its plain label. Plain text is
    returned unchanged."""
    s = text
    while s and not (s[0].isascii() and s[0].isalnum()):
        s = s[1:].lstrip()
    return s


# Button labels: plain English (for matching) + bold display text.
LBL_HELP = "Help"
LBL_CANCEL = "Cancel"
LBL_BROADCAST = "Broadcast"
LBL_STATS = "Stats"
LBL_OWNER = "Owner"
LBL_LANG = "Language"
LBL_T2V = "Text to Voice"
LBL_T2V_ON = "Text to Voice: ON"
LBL_T2V_OFF = "Text to Voice: OFF"
LBL_SHARE_LINK = "Share link"
LBL_LINK_LIST = "Link list"
LBL_REMOVE_LINK = "Remove link"
LBL_CREATE_LINK = "Create link"
LBL_FORCE_CREATE = "Force Create"
LBL_FORCE_JOIN = "Force Join"
LBL_FJ_REMOVE = "Remove"
LBL_FJ_LIST = "List"

BTN_HELP = "❓ " + to_bold(LBL_HELP)
BTN_BROADCAST = "📢 " + to_bold(LBL_BROADCAST)
BTN_STATS = "📊 " + to_bold(LBL_STATS)
BTN_OWNER = "👑 " + to_bold(LBL_OWNER)
BTN_LANG = "🌐 " + to_bold(LBL_LANG)
BTN_T2V_ON = "🔊 " + to_bold(LBL_T2V_ON)
BTN_T2V_OFF = "🔊 " + to_bold(LBL_T2V_OFF)
BTN_SHARE_LINK = "🔗 " + to_bold(LBL_SHARE_LINK)
BTN_LINK_LIST = "📋 " + to_bold(LBL_LINK_LIST)
BTN_REMOVE_LINK = "🗑️ " + to_bold(LBL_REMOVE_LINK)
BTN_CREATE_LINK = "➕ " + to_bold(LBL_CREATE_LINK)
BTN_SHARE_CANCEL = "❌ " + to_bold(LBL_CANCEL)
BTN_FORCE_CREATE = "🔐 " + to_bold(LBL_FORCE_CREATE)
BTN_FORCE_JOIN = "📌 " + to_bold(LBL_FORCE_JOIN)
BTN_FJ_REMOVE = "🗑️ " + to_bold(LBL_FJ_REMOVE)
BTN_FJ_LIST = "📋 " + to_bold(LBL_FJ_LIST)


def is_admin(user_id: int) -> bool:
    return bool(ADMIN_ID) and user_id == ADMIN_ID


def main_keyboard() -> ReplyKeyboardMarkup:
    # No AI Chat / Talk to Admin buttons (removed 2026-11-27 on
    # Abu's order): every user message goes straight to the admin.
    return ReplyKeyboardMarkup(
        [[KeyboardButton(BTN_HELP, style="primary"),
          KeyboardButton(BTN_OWNER, style="danger")],
         [KeyboardButton(BTN_LANG, style="primary")]],
        resize_keyboard=True)


def admin_keyboard() -> ReplyKeyboardMarkup:
    # The Text to Voice button always shows the current state.
    t2v = BTN_T2V_ON if admin_voice_enabled() else BTN_T2V_OFF
    return ReplyKeyboardMarkup(
        [[KeyboardButton(BTN_BROADCAST, style="primary"),
          KeyboardButton(BTN_STATS, style="success")],
         [KeyboardButton(t2v, style="primary")],
         [KeyboardButton(BTN_SHARE_LINK, style="primary")],
         [KeyboardButton(BTN_FORCE_CREATE, style="primary")]],
        resize_keyboard=True)


def share_link_keyboard() -> ReplyKeyboardMarkup:
    """Share-link submenu: Link list / Remove link / Create link /
    Cancel. Shown only to the admin after tapping Share link."""
    return ReplyKeyboardMarkup(
        [[KeyboardButton(BTN_LINK_LIST, style="primary"),
          KeyboardButton(BTN_REMOVE_LINK, style="primary")],
         [KeyboardButton(BTN_CREATE_LINK, style="primary")],
         [KeyboardButton(BTN_SHARE_CANCEL, style="danger")]],
        resize_keyboard=True)


def force_create_keyboard() -> ReplyKeyboardMarkup:
    """Force Create submenu: Force Join / Remove / List / Cancel.
    Shown only to the admin after tapping Force Create."""
    return ReplyKeyboardMarkup(
        [[KeyboardButton(BTN_FORCE_JOIN, style="primary"),
          KeyboardButton(BTN_FJ_REMOVE, style="primary")],
         [KeyboardButton(BTN_FJ_LIST, style="primary")],
         [KeyboardButton(BTN_SHARE_CANCEL, style="danger")]],
        resize_keyboard=True)


WELCOME_ADMIN = (
    "👑 <b>এডমিন প্যানেল</b>\n\n"
    "ইউজারের মেসেজ এভাবে আসবে:\n"
    "নাম / ইউজারনেম / চ্যাট আইডি / মেসেজ\n\n"
    "উত্তর দিতে সেই মেসেজে <b>রিপ্লাই</b> দিন —\n"
    "উত্তর সরাসরি সেই ইউজারের কাছে যাবে।\n"
    "অথবা ব্যবহার করুন: <code>/reply user_id মেসেজ</code>\n\n"
    "📢 ব্রডকাস্ট: <code>/broadcast মেসেজ</code>\n"
    "📊 পরিসংখ্যান: <code>/stats</code>\n"
    "শেয়ার লিংক: 🔗 𝐒𝐡𝐚𝐫𝐞 𝐥𝐢𝐧𝐤 বাটন — লিংক তালিকা, মুছুন, নতুন লিংক\n"
    "📁 মিডিয়া: <code>/mediastats</code> • <code>/mediasearch 00001</code> • <code>/mediadelete 00001</code>\n"
    "🔒 ফোর্স জয়েন: পুরো বট ব্যবহার করতে গ্রুপে জয়েন করতে হবে\n"
    "   (ডিফল্ট গ্রুপ কোডে সেট করা; data/force_join.json দিয়ে বদলানো যায়)\n"
    "🔐 ফোর্স ক্রিয়েট: 🔐 𝐅𝐨𝐫𝐜𝐞 𝐂𝐫𝐞𝐚𝐭𝐞 বাটন — ফোর্স জয়েন সেটআপ / তালিকা / মুছুন\n\n"
    "🪪 মালিকের পরিচয় / তথ্য (শুধু আপনি):\n"
    "• <code>/setintro আপনার পরিচয়</code>\n"
    "• <code>/addinfo তথ্য</code> — তথ্য যোগ করুন\n"
    "• <code>/listinfo</code> — সব তথ্য দেখুন\n"
    "• <code>/delinfo নম্বর</code> — তথ্য মুছুন\n"
    "• <code>/owner</code> — পরিচয় দেখুন\n"
    "• <code>/unblock user_id</code> — স্প্যাম ব্লক তুলুন"
)

HELP_TEXT = (
    "ℹ️ <b>সাহায্য</b>\n\n"
    "✍️ যা লিখবেন বা পাঠাবেন, সব সরাসরি এডমিনের কাছে যাবে —\n"
    "   এডমিন নিজে উত্তর দেবেন।\n"
    f"👑 <b>{BTN_OWNER}</b> — বটের মালিকের পরিচয় দেখবেন।\n"
    f"🌐 <b>{BTN_LANG}</b> — বাংলা / English / 中文 বদলান।"
)

HELP_TEXT_EN = (
    "ℹ️ <b>Help</b>\n\n"
    "✍️ Everything you write or send goes straight to the admin —\n"
    "   the admin will reply personally.\n"
    f"👑 <b>{BTN_OWNER}</b> — see the bot owner's identity.\n"
    f"🌐 <b>{BTN_LANG}</b> — switch between Bengali / English / 中文."
)

HELP_TEXT_ZH = (
    "ℹ️ <b>帮助</b>\n\n"
    "✍️ 您写下或发送的所有内容都会直接发给管理员 —\n"
    "   管理员会亲自回复。\n"
    f"👑 <b>{BTN_OWNER}</b> — 查看机器人主人的身份。\n"
    f"🌐 <b>{BTN_LANG}</b> — 切换孟加拉语 / 英语 / 中文。"
)


# ---------------- Full-bot translation (feature 14, extended) ----------------
# The Language button switches the WHOLE bot, not just AI answers:
# every user-facing message follows the user's own language choice
# (data/lang.json, bn/en/zh). Admin-facing strings stay in Bengali
# (the admin is Abu). Reply-keyboard button labels keep their
# English bold-Unicode form (Bengali has no bold form).
STRINGS = {
    "welcome": {
        "bn": (
            "🔥 <b>স্বাগতম!</b>\n\n"
            "✍️ যা লিখবেন বা পাঠাবেন (ছবি, ভয়েস, ফাইল), সব "
            "সরাসরি এডমিনের কাছে যাবে — এডমিন নিজে উত্তর দেবেন।\n"
            "{owner} চাপ দিলে বটের মালিকের পরিচয় দেখবেন।\n"
            "{lang} চাপ দিলে বাংলা / English / 中文 বদলাতে পারবেন।"),
        "en": (
            "🔥 <b>Welcome!</b>\n\n"
            "✍️ Everything you write or send (photos, voice, "
            "files) goes straight to the admin — the admin will "
            "reply personally.\n"
            "Tap {owner} to see the bot owner's identity.\n"
            "Tap {lang} to switch Bengali / English / 中文。"),
        "zh": (
            "🔥 <b>欢迎！</b>\n\n"
            "✍️ 您写下或发送的所有内容（照片、语音、文件）都会"
            "直接发给管理员 — 管理员会亲自回复。\n"
            "点击 {owner} 可查看机器人主人的身份。\n"
            "点击 {lang} 可切换孟加拉语 / 英语 / 中文。"),
    },
    "send_failed": {
        "bn": "❌ মেসেজ পাঠানো যায়নি। কিছুক্ষণ পর আবার চেষ্টা করুন।",
        "en": "❌ The message could not be sent. Please try again in a little while.",
        "zh": "❌ 消息发送失败。请稍后再试。",
    },
    "spam_blocked": {
        "bn": ("⛔ অতিরিক্ত মেসেজের কারণে আপনাকে ব্লক করা হয়েছে।\n"
               "এডমিনের সাথে যোগাযোগ করুন।"),
        "en": ("⛔ You have been blocked for sending too many messages.\n"
               "Please contact the admin."),
        "zh": ("⛔ 由于消息过多，您已被屏蔽。\n"
               "请联系管理员。"),
    },
    "error_retry": {
        "bn": "⚠️ একটু সমস্যা হয়েছে, আবার চেষ্টা করুন।",
        "en": "⚠️ Something went wrong, please try again.",
        "zh": "⚠️ 出了一点问题，请再试一次。",
    },
}


def t(key: str, lang: str = "bn") -> str:
    """One user-facing string in the user's language.

    {owner}, {lang} and {help} are filled with the bold button
    labels at call time.
    """
    text = STRINGS[key].get(lang) or STRINGS[key]["bn"]
    return text.format(owner=BTN_OWNER, lang=BTN_LANG, help=BTN_HELP)


def help_text(lang: str = "bn") -> str:
    if lang == "en":
        return HELP_TEXT_EN
    if lang == "zh":
        return HELP_TEXT_ZH
    return HELP_TEXT


# Admin-message translate store: token -> {text, user_id, full,
# to_lang, translated, shown}. In-memory only; a restart clears it,
# so old buttons just answer that the message is gone.
_ADMIN_MSG: dict = {}


def detect_msg_lang(text: str) -> str:
    """'bn' if the text contains Bengali script, else 'en'."""
    return "bn" if re.search(r"[\u0980-\u09FF]", text or "") else "en"


# ---------------- Multilingual: every language + Banglish ----------------
# (Abu, 2026-11-24): admin replies are auto-translated to the
# user's language, and Text-to-Voice must speak every language —
# including Banglish (Bengali written in Latin letters).
# Detection is script-based for non-Latin scripts; Latin text is
# checked for Banglish. Cartesia Sonic 3 speaks 40 languages
# (docs.cartesia.ai, Sonic 3.5 page); Urdu and Odia need the
# sonic-3.6 model, so those two switch models per call. A cloned
# voice speaks other languages with the same timbre — verified live
# 2026-11-24 (Abu Taher clone in Hindi and Arabic).
CARTESIA_LANGS = frozenset({
    "en", "de", "es", "fr", "it", "nl", "pl", "ru", "sv", "tr", "cs",
    "el", "fi", "hr", "sk", "da", "uk", "hu", "no", "pt", "ro", "bg",
    "zh", "ja", "ko", "vi", "th", "id", "ms", "tl", "ta", "te", "gu",
    "kn", "ml", "mr", "pa", "bn", "hi", "ar", "he", "ka"})
# Sonic 3.6-only languages (not on Sonic 3).
CARTESIA_LANGS_36 = frozenset({"ur", "or"})

_SCRIPT_RANGES = (
    ("bn", "ঀ", "৿"), ("hi", "ऀ", "ॿ"),
    ("ar", "؀", "ۿ"), ("ta", "஀", "௿"),
    ("te", "ఀ", "౿"), ("gu", "઀", "૿"),
    ("pa", "਀", "੿"), ("ml", "ഀ", "ൿ"),
    ("kn", "ಀ", "೿"), ("or", "଀", "୿"),
    ("th", "฀", "๿"), ("ja", "ぁ", "ヿ"),
    ("zh", "一", "鿿"), ("ko", "가", "힯"),
    ("ru", "Ѐ", "ӿ"), ("he", "֐", "׿"),
    ("ka", "Ⴀ", "ჿ"),
)
# Letters that appear in Urdu but not in Arabic.
_URDU_CHARS = set("ٹڈڑںھےۓیک")

# Common Banglish words -> correct Bengali. This dictionary is the
# main quality lever: it fixes words the phonetic engine would get
# wrong (টাকা, কেমন, কথা, ...) and doubles as the Banglish detector.
_BANGLISH_DICT = {
    "ami": "আমি", "tumi": "তুমি", "apni": "আপনি", "tui": "তুই",
    "tor": "তোর", "tar": "তার", "amar": "আমার", "tomar": "তোমার",
    "amader": "আমাদের", "tomader": "তোমাদের", "oder": "ওদের",
    "amake": "আমাকে", "tomake": "তোমাকে", "take": "তাকে",
    "she": "সে", "je": "যে", "ke": "কে",
    "kemon": "কেমন", "acho": "আছো", "achen": "আছেন",
    "achi": "আছি", "ache": "আছে", "nei": "নেই",
    "hobe": "হবে", "hobo": "হবো", "chilo": "ছিলো",
    "thakbe": "থাকবে", "holo": "হলো",
    "koro": "করো", "korbo": "করবো", "korchi": "করছি",
    "korchen": "করছেন", "korlam": "করলাম",
    "kora": "করা", "kore": "করে", "korte": "করতে",
    "korle": "করলে", "korbe": "করবে", "korar": "করার",
    "korechi": "করেছি", "koreche": "করেছে",
    "korecho": "করেছো", "korechen": "করেছেন",
    "jao": "যাও", "jacchi": "যাচ্ছি", "jabo": "যাবো",
    "jaoa": "যাওয়া", "jai": "যাই", "jaoar": "যাওয়ার",
    "jete": "যেতে", "jacche": "যাচ্ছে", "jachhi": "যাচ্ছি",
    "jabe": "যাবে", "jaben": "যাবেন", "jawa": "যাওয়া",
    "jay": "যায়", "jacchen": "যাচ্ছেন",
    "aschi": "আসছি", "asbo": "আসবো", "ashlam": "আসলাম",
    "asche": "আসছে", "ashche": "আসছে", "ashchen": "আসছেন",
    "eshechi": "এসেছি", "esheche": "এসেছে", "eshe": "এসে",
    "gechi": "গেছি", "geche": "গেছে",
    "bolechi": "বলেছি", "boleche": "বলেছে", "bolchen": "বলছেন",
    "dekhechi": "দেখেছি", "dekheche": "দেখেছে",
    "shunechi": "শুনেছি", "shunecho": "শুনেছো",
    "kheyechi": "খেয়েছি", "kheyecho": "খেয়েছো",
    "kheyechen": "খেয়েছেন",
    "gelam": "গেলাম", "khabo": "খাবো", "khaccho": "খাচ্ছো",
    "dao": "দাও", "deo": "দাও", "dibo": "দিবো", "diben": "দিবেন",
    "nibo": "নিবো", "nibe": "নিবে",
    "shuno": "শুনো", "shunlam": "শুনলাম",
    "bolo": "বলো", "bolen": "বলেন", "bolchi": "বলছি",
    "bollam": "বললাম", "dekhlam": "দেখলাম",
    "dekhchi": "দেখছি", "dekhbo": "দেখবো",
    "bhalo": "ভালো", "kharap": "খারাপ",
    "bhalobasha": "ভালোবাসা", "bondhu": "বন্ধু",
    "bhai": "ভাই", "bon": "বোন", "ma": "মা", "baba": "বাবা",
    "chele": "ছেলে", "meye": "মেয়ে",
    "dhonnobad": "ধন্যবাদ",
    "kotha": "কথা", "kothay": "কোথায়", "kothao": "কোথাও",
    "ekhane": "এখানে", "shekhane": "সেখানে", "okhane": "ওখানে",
    "ekhono": "এখনো", "kokhono": "কখনো", "kokhon": "কখন",
    "ki": "কি", "keno": "কেন", "kivabe": "কিভাবে",
    "kobe": "কবে", "kotokkhon": "কতক্ষণ",
    "koto": "কত", "kotota": "কতটা",
    "na": "না", "ha": "হ্যাঁ", "hya": "হ্যাঁ",
    "thik": "ঠিক", "kintu": "কিন্তু", "jodi": "যদি",
    "tahole": "তাহলে", "ekhon": "এখন",
    "pore": "পরে", "age": "আগে",
    "ajke": "আজকে", "aj": "আজ", "kalke": "কালকে", "kal": "কাল",
    "gotokal": "গতকাল", "porshu": "পরশু",
    "shokal": "সকাল", "shokale": "সকালে",
    "rat": "রাত", "rate": "রাতে",
    "dupur": "দুপুর", "dupure": "দুপুরে", "bikal": "বিকাল",
    "bangla": "বাংলা", "bangladesh": "বাংলাদেশ", "dhaka": "ঢাকা",
    "taka": "টাকা", "dam": "দাম",
    "onek": "অনেক", "kom": "কম", "beshi": "বেশি", "olpo": "অল্প",
    "choto": "ছোট", "boro": "বড়", "notun": "নতুন",
    "purono": "পুরনো", "shundor": "সুন্দর",
    "ashole": "আসলে", "shotti": "সত্যি", "obosshoi": "অবশ্যই",
    "asha": "আশা", "moshjid": "মসজিদ", "mosjid": "মসজিদ",
    "salam": "সালাম", "assalamu": "আসসালামু",
    "alaikum": "আলাইকুম", "waalaikum": "ওয়ালাইকুম",
    "allah": "আল্লাহ", "hafiz": "হাফেজ",
    "shukriya": "শুকরিয়া", "alhamdulillah": "আলহামদুলিল্লাহ",
    "mashallah": "মাশাআল্লাহ", "inshallah": "ইনশাআল্লাহ",
    "onno": "অন্য", "eita": "এইটা", "oita": "ওইটা",
    "ekta": "একটা", "duita": "দুইটা",
    "shobai": "সবাই", "shob": "সব", "kichu": "কিছু",
    "keu": "কেউ", "kono": "কোনো", "konta": "কোনটা",
    "kache": "কাছে", "theke": "থেকে", "diye": "দিয়ে",
    "jonno": "জন্য", "shathe": "সাথে",
    "moddhe": "মধ্যে", "upore": "উপরে", "niche": "নিচে",
    "bhitore": "ভিতরে", "baire": "বাইরে",
    "chai": "চাই", "lagbe": "লাগবে",
    "parbo": "পারবো", "parbe": "পারবে",
    "janina": "জানিনা", "janen": "জানেন",
    "bujhi": "বুঝি", "bujhlam": "বুঝলাম",
    "kanya": "কন্যা", "kahini": "কাহিনী",
    "kosto": "কষ্ট", "koshto": "কষ্ট",
    "golpo": "গল্প", "gan": "গান", "chobi": "ছবি",
    "khela": "খেলা", "kaj": "কাজ", "bari": "বাড়ি",
    "basha": "বাসা", "rasta": "রাস্তা",
    "dokan": "দোকান", "bazar": "বাজার",
    "pani": "পানি", "bhat": "ভাত", "mach": "মাছ",
    "mangsho": "মাংস", "shobji": "সবজি", "misti": "মিষ্টি",
    "cha": "চা", "doodh": "দুধ", "chini": "চিনি",
    "lobon": "লবণ", "morich": "মরিচ", "holud": "হলুদ",
    "adha": "আদা", "roshun": "রসুন", "piyaj": "পেঁয়াজ",
    "alu": "আলু", "begun": "বেগুন", "tomato": "টমেটো",
    "kacha": "কাঁচা", "paka": "পাকা",
    "gorom": "গরম", "thanda": "ঠান্ডা",
    "brishti": "বৃষ্টি", "rod": "রোদ", "megh": "মেঘ",
    "akash": "আকাশ", "chand": "চাঁদ", "surjo": "সূর্য",
    "tara": "তারা", "mati": "মাটি", "gach": "গাছ",
    "phul": "ফুল", "pakhi": "পাখি",
    "kukur": "কুকুর", "biral": "বিড়াল", "goru": "গরু",
    "chagol": "ছাগল", "shap": "সাপ",
    "hat": "হাত", "pa": "পা", "chokh": "চোখ",
    "kan": "কান", "nak": "নাক", "mukh": "মুখ",
    "matha": "মাথা", "chul": "চুল", "dat": "দাঁত",
    "shorir": "শরীর", "mon": "মন",
    "khushi": "খুশি", "dukkho": "দুঃখ", "rag": "রাগ",
    "bhoy": "ভয়", "asha": "আশা", "shopno": "স্বপ্ন",
    "jibon": "জীবন", "moron": "মরণ", "jonmo": "জন্ম",
    "biye": "বিয়ে", "baccha": "বাচ্চা", "shishu": "শিশু",
    "chatro": "ছাত্র", "shikkhok": "শিক্ষক",
    "school": "স্কুল", "college": "কলেজ",
    "boi": "বই", "kolom": "কলম", "khata": "খাতা",
    "pora": "পড়া", "lekha": "লেখা", "shikha": "শেখা",
    "jana": "জানা", "bojha": "বোঝা", "mone": "মনে",
}
# Word endings that often mark Banglish. They only count when at
# least one dictionary word is present, so English text like
# "after dinner" (which ends in the same letters) is not mistaken
# for Banglish.
_BANGLISH_SUFFIXES = (
    "bo", "be", "lam", "le", "lo", "cho", "che", "chen", "chi",
    "ar", "er", "ke", "te", "ta", "na", "ni", "to", "ti", "ra",
    "re", "der", "gulo", "tai", "kor", "kore", "korbo", "korchi",
    "hobe", "ache", "acho", "achen", "theke", "diye", "jonno",
    "por", "age", "ekhon", "shathe",
)


def is_banglish(text: str) -> bool:
    """True when Latin-script text reads as Banglish (Romanized
    Bengali) rather than English."""
    words = re.findall(r"[A-Za-z]+", text or "")
    if not words:
        return False
    dict_hits = sum(1 for w in words if w.lower() in _BANGLISH_DICT)
    if not dict_hits:
        return False
    suffix_hits = sum(
        0.5 for w in words
        if len(w) >= 3 and w.lower().endswith(_BANGLISH_SUFFIXES))
    score = dict_hits + suffix_hits
    return dict_hits >= 2 or score / len(words) >= 0.3


def detect_text_lang(text: str) -> str:
    """Detect a message's language for the AI and TTS.

    Returns a Cartesia/AI language code: 'bn' for Bengali script,
    'banglish' for Romanized Bengali, 'en' for English, 'hi' for
    Devanagari (Hindi/Marathi/Nepali share the script — read as
    Hindi), 'ar'/'ur' for Arabic script, and the script's code for
    Tamil, Telugu, Gujarati, Punjabi, Malayalam, Kannada, Odia,
    Thai, Japanese, Chinese, Korean, Russian, Hebrew, Georgian.
    Unknown text defaults to 'en'."""
    t = text or ""
    counts: dict[str, int] = {}
    for ch in t:
        for code, lo, hi in _SCRIPT_RANGES:
            if lo <= ch <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    best = max(counts, key=counts.get) if counts else None
    if best and counts[best] >= 2:
        if best == "ar" and any(c in _URDU_CHARS for c in t):
            return "ur"
        return best
    return "banglish" if is_banglish(t) else "en"


# Language names for AI system prompts ("Reply in ...").
_LANG_NAMES = {
    "bn": "Bengali", "banglish": "Banglish (Bengali written in "
    "Latin/Roman letters — reply using Latin letters, not Bengali "
    "script)", "en": "English", "hi": "Hindi", "ar": "Arabic",
    "ur": "Urdu", "ta": "Tamil", "te": "Telugu", "gu": "Gujarati",
    "pa": "Punjabi", "ml": "Malayalam", "kn": "Kannada",
    "mr": "Marathi", "or": "Odia", "th": "Thai", "ja": "Japanese",
    "zh": "Chinese", "ko": "Korean", "ru": "Russian",
    "fr": "French", "de": "German", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch",
    "pl": "Polish", "sv": "Swedish", "tr": "Turkish",
    "tl": "Tagalog", "bg": "Bulgarian", "ro": "Romanian",
    "cs": "Czech", "el": "Greek", "fi": "Finnish",
    "hr": "Croatian", "ms": "Malay", "sk": "Slovak",
    "da": "Danish", "uk": "Ukrainian", "hu": "Hungarian",
    "no": "Norwegian", "vi": "Vietnamese", "he": "Hebrew",
    "ka": "Georgian", "id": "Indonesian",
}


def _reply_lang_name(lang: str) -> str:
    """English name of a language code for an AI prompt."""
    return _LANG_NAMES.get(lang or "bn", "English")


# ---- Banglish -> Bengali transliteration (phonetic, Avro-style) ----
# Vowels: independent form (word start / after a vowel) and the
# vowel sign used after a consonant. "a" after a consonant is the
# আ-কার — "bhalo" -> ভালো, "tomar" -> তোমার, "kora" -> করা.
_VOWELS = {"rri": "ঋ", "aa": "আ", "ii": "ঈ", "uu": "ঊ",
           "oi": "ঐ", "ou": "ঔ", "a": "আ", "i": "ই",
           "u": "উ", "e": "এ", "o": "ও"}
_SIGNS = {"rri": "ৃ", "aa": "া", "ii": "ী", "uu": "ূ",
          "oi": "ৈ", "ou": "ৌ", "a": "া", "i": "ি",
          "u": "ু", "e": "ে", "o": "ো"}
# Consonants, longest match first. Doubled letters and common
# clusters are single units so the hasanta rule renders them as
# conjuncts (ক্ক, স্ত, ন্দ, ...). "ng" before "l" is the anusvara
# (বাংলা), otherwise ঙ. Words are lowercased first — capital
# T/D/N/S/R distinctions are unreliable in Banglish, so the
# dictionary carries words like টাকা and ঠিক.
_CONS = {
    "sht": "ষ্ট", "ngk": "ঙ্ক", "ngg": "ঙ্গ", "nch": "ঞ্চ",
    "shr": "শ্র", "chh": "ছ",
    "kh": "খ", "gh": "ঘ", "ch": "ছ", "jh": "ঝ", "th": "থ",
    "dh": "ধ", "ph": "ফ", "bh": "ভ", "sh": "শ", "ng": "ঙ",
    "ny": "ঞ", "st": "স্ত", "nd": "ন্দ", "nt": "ন্ত",
    "mp": "ম্প", "mb": "ম্ব", "kk": "ক্ক", "tt": "ট্ট",
    "ll": "ল্ল", "ss": "স্স", "nn": "ন্ন", "mm": "ম্ম",
    "dd": "দ্দ", "gg": "গ্গ", "bb": "ব্ব", "jj": "জ্জ",
    "kr": "ক্র", "pr": "প্র", "br": "ব্র", "tr": "ত্র",
    "gr": "গ্র", "dr": "দ্র", "kl": "ক্ল", "sk": "স্ক",
    "sp": "স্প", "sm": "স্ম", "sn": "স্ন", "sw": "স্ব",
    "tw": "ত্ব", "kw": "ক্ব", "ky": "ক্য", "gy": "গ্য",
    "by": "ব্য", "my": "ম্য", "py": "প্য", "dy": "দ্য",
    "ty": "ত্য",
    "k": "ক", "g": "গ", "c": "চ", "j": "জ", "t": "ত",
    "d": "দ", "n": "ন", "p": "প", "f": "ফ", "b": "ব",
    "v": "ভ", "m": "ম", "r": "র", "l": "ল", "s": "স",
    "h": "হ", "z": "জ", "q": "ক", "x": "ক্স",
}


def _transliterate_word(word: str) -> str:
    """One Banglish word -> Bengali script (best effort)."""
    w = word.lower()
    if w in _BANGLISH_DICT:
        return _BANGLISH_DICT[w]
    out: list[str] = []
    i, n = 0, len(w)
    last = "start"  # "start" | "cons" | "vowel"
    while i < n:
        ch = w[i]
        if ch == "y":
            # Word-final y is য়; between two vowels it is the য়
            # glide (kheyechi -> খেয়েছি); elsewhere য (conjunct
            # after a consonant, e.g. kya -> ক্যা).
            if i == n - 1:
                out.append("য়")
                last = "cons"
            elif (last == "vowel" and i + 1 < n
                    and w[i + 1] in "aeiou"):
                # য় takes vowel signs after it like a consonant
                # (kheyechi -> খেয়েছি), so it counts as "cons".
                out.append("য়")
                last = "cons"
            else:
                if last == "cons":
                    out.append("্")
                out.append("য")
                last = "cons"
            i += 1
            continue
        if ch == "w":
            # After a vowel it is the উ sound (kew -> কেউ).
            if last == "vowel":
                out.append("উ")
                last = "vowel"
            elif last == "start":
                out.append("ও")
                last = "vowel"
            else:
                out.append("্")
                out.append("ও")
                last = "cons"
            i += 1
            continue
        cons = None
        for size in (3, 2, 1):
            chunk = w[i:i + size]
            if chunk in _CONS:
                cons = (chunk, _CONS[chunk])
                break
        if cons:
            chunk, beng = cons
            if chunk == "ng" and i + 2 < n and w[i + 2] == "l":
                out.append("ং")
                last = "vowel"
                i += 2
                continue
            if last == "cons":
                out.append("্")
            out.append(beng)
            last = "cons"
            i += len(chunk)
            continue
        vow = None
        for size in (3, 2, 1):
            chunk = w[i:i + size]
            if chunk in _VOWELS:
                vow = chunk
                break
        if vow:
            if last == "cons":
                out.append(_SIGNS[vow])
            elif last == "vowel" and vow == "a":
                # "a" after a vowel is the য়া glide
                # (jaoa -> যাওয়া, khaoa -> খাওয়া).
                out.append("য়া")
            else:
                out.append(_VOWELS[vow])
            last = "vowel"
            i += len(vow)
            continue
        # Unknown character (apostrophe, digit, ...): copy through.
        out.append(ch)
        i += 1
    return "".join(out)


def banglish_to_bengali(text: str) -> str:
    """Transliterate Latin-script Banglish text to Bengali script."""
    return re.sub(r"[A-Za-z]+(?:'[A-Za-z]+)?",
                  lambda m: _transliterate_word(m.group(0)),
                  text or "")


def prepare_tts_text(text: str) -> tuple[str, str]:
    """(cartesia_lang, speak_text) for TTS. Banglish is
    transliterated to Bengali script and spoken as bn; every other
    detected language passes its own text through with its Cartesia
    code. Unknown text defaults to English."""
    lang = detect_text_lang(text)
    if lang == "banglish":
        return "bn", banglish_to_bengali(text)
    if lang in CARTESIA_LANGS or lang in CARTESIA_LANGS_36:
        return lang, text
    return "en", text


def _admin_msg_markup(token: str, entry: dict) -> InlineKeyboardMarkup:
    """Two Translate buttons (Bn / En) for one admin message.

    Tapping a language shows its translation below the original;
    tapping the same language again hides it; tapping the other
    language switches the translation. The original always stays."""
    shown = entry.get("shown_lang")
    bn_label = "🇧🇩 " + to_bold("Bn") + (" ✅" if shown == "bn" else "")
    en_label = "🇬🇧 " + to_bold("En") + (" ✅" if shown == "en" else "")
    return InlineKeyboardMarkup([[
        InlineKeyboardButton(bn_label, callback_data=f"tr:{token}:bn",
                             style="primary"),
        InlineKeyboardButton(en_label, callback_data=f"tr:{token}:en",
                             style="primary"),
    ]])


def translate_text(text: str, target_lang: str) -> str | None:
    """Translate a message (Groq first, Pollinations fallback).
    Any language code works (bn/en/zh/hi/ar/.../banglish); the
    target name comes from _reply_lang_name. Short translate-only
    prompt, NOT a full AI answer. Returns None on any failure."""
    target = _reply_lang_name(target_lang)
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


async def _callback_force_join_blocked(context, query) -> bool:
    """Gate for user-facing inline callbacks (language):
    a non-member gets a toast plus the join prompt as a new
    message, and the callback does nothing. True = blocked."""
    user = query.from_user
    if not user or is_admin(user.id):
        return False
    if await force_join_check(context, user.id) != "block":
        return False
    try:
        await query.answer("গ্রুপে জয়েন করে আবার চেষ্টা করুন।")
    except Exception:
        pass
    try:
        chat_id = query.message.chat_id if query.message else user.id
        await context.bot.send_message(
            chat_id=chat_id, text=FJ_PROMPT_TEXT,
            parse_mode=ParseMode.HTML,
            reply_markup=force_join_markup("start"))
    except Exception:
        log.warning("force join prompt failed", exc_info=True)
    return True


async def lang_callback(update: Update,
                        context: ContextTypes.DEFAULT_TYPE) -> None:
    """Feature 14 (corrected): language chosen by inline button tap."""
    query = update.callback_query
    if not query or not query.data:
        return
    if await _callback_force_join_blocked(context, query):
        return
    try:
        _, lang = query.data.split(":", 1)
    except ValueError:
        await query.answer()
        return
    if lang not in ("bn", "en", "zh"):
        await query.answer()
        return
    user = query.from_user
    set_lang(user.id, lang)
    if lang == "en":
        text = ("Language set to English. 🇬🇧\n"
                "The whole bot will now speak English.")
        toast = "English selected"
    elif lang == "zh":
        text = ("语言已设置为中文。🇨🇳\n"
                "整个机器人现在将使用中文。")
        toast = "已选择中文"
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
    """Admin Translate buttons (Bn / En): show the translation of
    one user message below the original. Tapping the same language
    again hides it; tapping the other switches it. The original
    text always stays."""
    query = update.callback_query
    if not query or not query.data:
        return
    parts = query.data.split(":", 1)
    if len(parts) < 2:
        await query.answer()
        return
    rest = parts[1]
    # Tokens themselves contain a colon (user_id:timestamp), so the
    # language suffix is stripped from the END: tr:<token>:bn.
    if rest.endswith(":bn") or rest.endswith(":en"):
        token, want = rest[:-3], rest[-2:]
    else:
        # Old single button sent tr:<token> (falls back to the
        # entry's to_lang).
        token, want = rest, None
    entry = _ADMIN_MSG.get(token)
    if not entry:
        await query.answer("মেসেজটি আর নেই।")
        return
    if want not in ("bn", "en"):
        want = entry.get("to_lang") or "bn"
    if entry.get("shown_lang") == want:
        # Same language again -> hide the translation.
        entry["shown_lang"] = None
        new_text = entry["full"]
        await query.answer("মূল মেসেজ দেখানো হয়েছে।")
    else:
        translated = entry.get("translated") or {}
        if want not in translated:
            got = await asyncio.to_thread(
                translate_text, entry["text"], want)
            if not got:
                await query.answer("অনুবাদ করা যায়নি।")
                return
            translated[want] = got
            entry["translated"] = translated
        target = "English" if want == "en" else "বাংলা"
        new_text = (entry["full"] + f"\n\n🌐 অনুবাদ ({target}):\n"
                    + translated[want])
        entry["shown_lang"] = want
        await query.answer("অনুবাদ দেখানো হয়েছে।")
    try:
        await query.edit_message_text(
            new_text, reply_markup=_admin_msg_markup(token, entry))
    except Exception as e:
        log.warning("translate edit failed: %s", e)


# ---------------- Voice replies / TTS ----------------
# TTS is used for the admin's Text-to-Voice toggle
# (send_admin_reply_as_voice) only.
# Voice comes FIRST from Abu's own Cartesia voice clone (he made it himself; account holds an owned Bengali
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
    """Multilingual TTS via Abu's Cartesia clone voice.

    Sonic 3 speaks 40 languages; Urdu and Odia switch to the
    sonic-3.6 model. Returns WAV bytes (pcm_s16le, 44100 Hz, mono)
    or None on any failure so the caller falls back.
    """
    if not CARTESIA_AVAILABLE:
        return None
    lang = lang if (lang in CARTESIA_LANGS
                    or lang in CARTESIA_LANGS_36) else "en"
    model = "sonic-3.6" if lang in CARTESIA_LANGS_36 else CARTESIA_MODEL
    try:
        import json
        import urllib.error
        body = json.dumps({
            "model_id": model,
            "transcript": text,
            "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
            "language": lang,
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


def make_voice_bytes(text: str, lang: str | None = None) -> bytes | None:
    """Multilingual TTS -> Opus/OGG bytes. Returns None on failure.

    With no lang given, the text's own language is detected
    (Banglish is transliterated to Bengali script first). Cartesia
    speaks 40+ languages with Abu's clone voice; the local tts
    binary only covers Bengali/English, so for other languages a
    Cartesia failure means no voice (the caller sends text)."""
    clean = re.sub(r"<[^>]+>", "", text or "").strip()
    clean = clean[:600]
    if not clean:
        return None
    if lang is None:
        lang, clean = prepare_tts_text(clean)
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
            log.info("voice via cartesia clone (%s)", lang)
        elif lang not in ("bn", "en"):
            # Local tts only speaks Bengali/English — a wrong-
            # language voice is worse than no voice at all.
            log.warning("tts skipped: no local voice for %s", lang)
            return None
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
    in the reply's own language (every language, Banglish included).
    Falls back to sending the text if TTS fails. Returns True if
    voice was sent."""
    data = None
    try:
        data = await asyncio.to_thread(make_voice_bytes, text)
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


async def maybe_translate_reply(text: str,
                                target_lang: str) -> tuple[str, bool]:
    """Auto-translate the admin's reply to the user's language
    (Abu, 2026-11-24). Returns (text_to_send, was_translated).
    Same language -> the original untouched; a failed translate
    never blocks the reply (the original goes out)."""
    if not text or not text.strip():
        return text, False
    target = target_lang or "bn"
    if detect_text_lang(text) == target:
        return text, False
    translated = await asyncio.to_thread(translate_text, text, target)
    if not translated:
        return text, False
    log.info("admin reply auto-translated to %s", target)
    return translated, True


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
                  f"মেসেজ👇\n\n{(text or '')[:500]}"),
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
            # Photo (with or without a caption). Telegram captions
            # cap at 1024 chars, so a long caption goes as a
            # follow-up text instead of being silently cut.
            full_caption = (header + "\nমেসেজ👇\n\n"
                            + (text or "📷 ছবি পাঠিয়েছে"))
            if len(full_caption) <= 1000:
                sent = await context.bot.send_photo(
                    chat_id=ADMIN_ID, photo=msg.photo[-1].file_id,
                    caption=full_caption)
                sent_ids.append(sent.message_id)
            else:
                sent = await context.bot.send_photo(
                    chat_id=ADMIN_ID, photo=msg.photo[-1].file_id,
                    caption=header + "\nমেসেজ👇\n\n📷 ছবি পাঠিয়েছে")
                sent_ids.append(sent.message_id)
                sent2 = await context.bot.send_message(
                    chat_id=ADMIN_ID, text=full_caption)
                sent_ids.append(sent2.message_id)
        elif (msg.video or msg.document or msg.animation or msg.audio
              or msg.voice or msg.sticker or msg.video_note):
            # Media with or without a caption: the media itself must
            # reach the admin. The caption travels with the copied
            # media (copy_message keeps it) — before this fix, a
            # captioned video/file only sent its caption as text.
            if msg.voice:
                kind = "(ভয়েস পাঠিয়েছে)"
            elif msg.video:
                kind = "(ভিডিও পাঠিয়েছে)"
            elif msg.document:
                kind = "(ফাইল পাঠিয়েছে)"
            elif msg.sticker:
                kind = "(স্টিকার পাঠিয়েছে)"
            elif msg.animation:
                kind = "(GIF পাঠিয়েছে)"
            elif msg.audio:
                kind = "(অডিও পাঠিয়েছে)"
            elif msg.video_note:
                kind = "(ভিডিও নোট পাঠিয়েছে)"
            else:
                kind = "(ছবি/ফাইল/স্টিকার পাঠিয়েছে)"
            sent = await context.bot.send_message(
                chat_id=ADMIN_ID,
                text=header + "\nমেসেজ👇\n\n" + kind)
            sent_ids.append(sent.message_id)
            copied = await context.bot.copy_message(
                chat_id=ADMIN_ID, from_chat_id=msg.chat_id,
                message_id=msg.message_id)
            sent_ids.append(copied.message_id)
        elif text:
            token = f"{user.id}:{int(time.time())}"
            full = header + "\nমেসেজ👇\n\n" + text
            to_lang = "en" if detect_msg_lang(text) == "bn" else "bn"
            if len(_ADMIN_MSG) > 500:  # keep the in-memory store bounded
                for _k in list(_ADMIN_MSG)[:len(_ADMIN_MSG) - 500]:
                    _ADMIN_MSG.pop(_k, None)
            _ADMIN_MSG[token] = {"text": text, "user_id": user.id,
                                 "full": full, "to_lang": to_lang,
                                 "translated": {}, "shown_lang": None}
            sent = await context.bot.send_message(
                chat_id=ADMIN_ID, text=full,
                reply_markup=_admin_msg_markup(token, _ADMIN_MSG[token]))
            sent_ids.append(sent.message_id)
        else:
            # Anything else (contact, location, ...): header + copy.
            sent = await context.bot.send_message(
                chat_id=ADMIN_ID,
                text=header + "\nমেসেজ👇\n\n"
                + "(ছবি/ফাইল/স্টিকার পাঠিয়েছে)")
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


# ---------------- handlers ----------------

async def send_user_welcome(context, chat_id: int, user) -> None:
    """The normal plain-/start welcome for a non-admin user.
    Also used after a successful 'start'-context force-join
    verify, so the welcome is identical in both paths."""
    DATA["users"].setdefault(str(user.id), {
        "name": user.full_name,
        "username": user.username or "",
    })
    save_data(DATA)
    lang = get_lang(user.id)
    await context.bot.send_message(
        chat_id=chat_id, text=t("welcome", lang),
        parse_mode=ParseMode.HTML,
        reply_markup=main_keyboard())


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    # Deep link: /start 00001 -> deliver the stored media to anyone.
    # No argument -> the usual welcome / admin panel below.
    args = context.args or []
    if args:
        code = args[0].strip()
        if not is_valid_media_code(code):
            await update.message.reply_text(
                "Sorry, this file could not be found.")
            return
        try:
            row = media_get_by_code(code)
        except Exception:
            log.exception("deep link lookup failed")
            row = None
        if row is None:
            await update.message.reply_text(
                "Sorry, this file could not be found.")
            return
        # Force join: non-admin users must be group members first.
        # The verify context is the media code, so a successful
        # verify delivers this file.
        if not is_admin(user.id):
            gate = await force_join_check(context, user.id)
            if gate == "block":
                await update.message.reply_text(
                    FJ_PROMPT_TEXT, parse_mode=ParseMode.HTML,
                    reply_markup=force_join_markup(code))
                return
        ok = await send_stored_media(context, update.effective_chat.id,
                                     row)
        if not ok:
            await update.message.reply_text(
                "Sorry, this file could not be found.")
        return
    if is_admin(user.id):
        await update.message.reply_text(
            WELCOME_ADMIN, parse_mode=ParseMode.HTML,
            reply_markup=admin_keyboard())
        return
    # Global force join: a plain /start is gated the same way as a
    # deep link — non-members get the join prompt, not the welcome.
    gate = await force_join_check(context, user.id)
    if gate == "block":
        await update.message.reply_text(
            FJ_PROMPT_TEXT, parse_mode=ParseMode.HTML,
            reply_markup=force_join_markup("start"))
        return
    await send_user_welcome(context, update.effective_chat.id, user)


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
    # --- Share-link submenu: checked FIRST, so its Cancel returns to
    # the admin keyboard and is never swallowed by the generic
    # handlers below. ---
    if share_menu_active() or share_pending() is not None:
        if plain == LBL_SHARE_LINK:
            await msg.reply_text(
                "🔗 <b>শেয়ার লিংক</b>",
                parse_mode=ParseMode.HTML,
                reply_markup=share_link_keyboard())
            return
        if plain == LBL_LINK_LIST:
            await show_link_list(msg)
            return
        if plain == LBL_REMOVE_LINK:
            set_share_state(True, "remove")
            await msg.reply_text(
                "🗑️ <b>লিংক মুছুন</b>\n\n"
                "মুছতে চাওয়া ফাইলের File ID লিখুন "
                "(যেমন <code>00001</code>)।",
                parse_mode=ParseMode.HTML,
                reply_markup=share_link_keyboard())
            return
        if plain == LBL_CREATE_LINK:
            set_share_state(True, "create")
            await msg.reply_text(
                "🔗 <b>লিংক তৈরি</b>\n\n"
                "মিডিয়া পাঠান — ছবি, ভিডিও, ডকুমেন্ট, অডিও, "
                "ভয়েস, GIF বা স্টিকার।\n"
                "একটি মিডিয়ায় একটি লিংক তৈরি হবে।",
                parse_mode=ParseMode.HTML,
                reply_markup=share_link_keyboard())
            return
        if plain == LBL_CANCEL:
            clear_share_state()
            await msg.reply_text(
                "↩️ এডমিন প্যানেলে ফিরেছেন।",
                reply_markup=admin_keyboard())
            return
        if share_pending() == "create":
            media = extract_media(msg)
            if media is not None:
                set_share_state(True, None)
                await handle_create_link_upload(update, context, msg,
                                                media)
                return
            # Not media: drop the create wait and let the message
            # flow through the normal admin handling below.
            set_share_state(True, None)
        elif share_pending() == "remove":
            code = (text or "").strip()
            if is_valid_media_code(code):
                set_share_state(True, None)
                status = await delete_media_by_code(context, code)
                if status == "ok":
                    await msg.reply_text(
                        f"🗑️ ফাইল <code>{code}</code> মুছে ফেলা হয়েছে।",
                        parse_mode=ParseMode.HTML,
                        reply_markup=share_link_keyboard())
                elif status == "not_found":
                    await msg.reply_text(
                        "Sorry, this file could not be found.",
                        reply_markup=share_link_keyboard())
                else:
                    await msg.reply_text(
                        "❌ মুছা যায়নি, আবার চেষ্টা করুন।",
                        reply_markup=share_link_keyboard())
                return
            await msg.reply_text(
                "সঠিক File ID লিখুন (যেমন <code>00001</code>)।",
                parse_mode=ParseMode.HTML,
                reply_markup=share_link_keyboard())
            return
        else:
            # In the submenu but the message is something else
            # (e.g. a reply to a user): leave the submenu and let
            # the normal admin handling below take it.
            clear_share_state()
    # --- Force Create submenu: checked FIRST (with the share-link
    # submenu) so its Cancel returns to the admin keyboard. ---
    if (force_menu_active() or force_setup_wait()
            or force_pending() is not None):
        if plain == LBL_FORCE_CREATE:
            set_force_state(True, False)
            await msg.reply_text(
                "🔐 <b>ফোর্স ক্রিয়েট</b>\n\n"
                "ফোর্স জয়েন সেট করতে, দেখতে বা মুছতে "
                "নিচের বাটন ব্যবহার করুন।",
                parse_mode=ParseMode.HTML,
                reply_markup=force_create_keyboard())
            return
        if plain == LBL_FORCE_JOIN:
            set_force_state(True, True)
            await msg.reply_text(
                "📌 <b>ফোর্স জয়েন সেটআপ</b>\n\n"
                "ফোর্স জয়েনের গ্রুপ/চ্যানেলের যেকোনো মেসেজ "
                "এখানে ফরওয়ার্ড করুন — অথবা @username বা "
                "-100 ID পাঠান।",
                parse_mode=ParseMode.HTML,
                reply_markup=force_create_keyboard())
            return
        if plain == LBL_FJ_REMOVE:
            cfg = load_force_join()
            if not cfg.get("chat_id"):
                await msg.reply_text(
                    "কোনো ফোর্স জয়েন সেট করা নেই।",
                    reply_markup=force_create_keyboard())
                return
            if cfg.get("from_default"):
                # Abu's rule (2026-11-27): the default force group can
                # NEVER be removed — do not even ask.
                await msg.reply_text(
                    "🔒 <b>ডিফল্ট ফোর্স জয়েন কখনো বাদ দেওয়া যাবে না।</b>\n\n"
                    f"গ্রুপ: {cfg.get('title') or '—'}\n"
                    f"চ্যাট আইডি: <code>{cfg['chat_id']}</code>",
                    parse_mode=ParseMode.HTML,
                    reply_markup=force_create_keyboard())
                return
            src_note = ("ডিফল্ট (কোডে সেট করা)"
                        if cfg.get("from_default") else "সেট করা")
            markup = InlineKeyboardMarkup([[
                InlineKeyboardButton(
                    "✅ Yes", callback_data="fjremove:yes",
                    style="danger"),
                InlineKeyboardButton(
                    "❌ No", callback_data="fjremove:no",
                    style="primary"),
            ]])
            await msg.reply_text(
                "🗑️ <b>ফোর্স জয়েন মুছবেন?</b>\n\n"
                f"গ্রুপ: {cfg.get('title') or '—'}\n"
                f"চ্যাট আইডি: <code>{cfg['chat_id']}</code>\n"
                f"উৎস: {src_note}\n"
                f"বাটন: {html.escape(cfg.get('button_name') or FORCE_JOIN_DEFAULT_BUTTON)}\n\n"
                "সত্যিই মুছতে চান?",
                parse_mode=ParseMode.HTML,
                reply_markup=markup)
            return
        if plain == LBL_FJ_LIST:
            cfg = load_force_join()
            if cfg.get("chat_id"):
                src_note = ("ডিফল্ট (কোডে সেট করা)"
                            if cfg.get("from_default") else "সেট করা")
                await msg.reply_text(
                    "📌 <b>ফোর্স জয়েন</b>\n\n"
                    f"গ্রুপ: {cfg.get('title') or '—'}\n"
                    f"চ্যাট আইডি: <code>{cfg['chat_id']}</code>\n"
                    f"উৎস: {src_note}\n"
                    f"বাটন: {html.escape(cfg.get('button_name') or FORCE_JOIN_DEFAULT_BUTTON)}\n"
                    f"লিংক: {cfg.get('bot_invite_link') or cfg.get('invite_link') or FORCE_JOIN_INVITE}",
                    parse_mode=ParseMode.HTML,
                    reply_markup=force_create_keyboard())
            else:
                await msg.reply_text(
                    "কোনো ফোর্স জয়েন সেট করা নেই।",
                    reply_markup=force_create_keyboard())
            return
        if plain == LBL_CANCEL:
            clear_force_state()
            await msg.reply_text(
                "↩️ এডমিন প্যানেলে ফিরেছেন।",
                reply_markup=admin_keyboard())
            return
        if force_pending() == "button_name":
            name = (text or "").strip()
            if name and len(name) <= 30:
                cfg = load_force_join()
                cfg["button_name"] = name
                cfg["pending"] = None
                save_force_join(cfg)
                await msg.reply_text(
                    "✅ <b>বাটনের নাম সেট হয়েছে</b>\n\n"
                    f"জয়েন বাটন: {html.escape(name)}",
                    parse_mode=ParseMode.HTML,
                    reply_markup=force_create_keyboard())
                log.info("force join: button name set to %s", name)
                return
            if not name:
                await msg.reply_text(
                    "জয়েন বাটনে কী নাম দেখাবে? নাম লিখে পাঠান।",
                    reply_markup=force_create_keyboard())
            else:
                await msg.reply_text(
                    "নামটি ৩০ অক্ষরের মধ্যে রাখুন।",
                    reply_markup=force_create_keyboard())
            return
        if force_setup_wait():
            if plain in (LBL_SHARE_LINK, LBL_FORCE_CREATE,
                         LBL_BROADCAST, LBL_STATS,
                         LBL_T2V_ON, LBL_T2V_OFF, LBL_T2V):
                # Leaving setup via another top-level admin button:
                # drop the wait and let normal handling take it
                # (like the Share-link create wait drops its
                # pending state for non-media messages).
                clear_force_state()
            elif await handle_force_setup_input(update, context, msg,
                                                text):
                return
            else:
                clear_force_state()
                # Fall through to normal admin handling below.
        else:
            # In the submenu but the message is something else
            # (e.g. a reply to a user): leave the submenu and let
            # the normal admin handling below take it.
            clear_force_state()
    if plain == LBL_FORCE_CREATE:
        clear_share_state()
        set_force_state(True, False)
        await msg.reply_text(
            "🔐 <b>ফোর্স ক্রিয়েট</b>\n\n"
            "ফোর্স জয়েন সেট করতে, দেখতে বা মুছতে "
            "নিচের বাটন ব্যবহার করুন।",
            parse_mode=ParseMode.HTML,
            reply_markup=force_create_keyboard())
        return
    if plain == LBL_SHARE_LINK:
        ADMIN_FLAGS["broadcast_pending"] = False
        set_share_state(True, None)
        await msg.reply_text(
            "🔗 <b>শেয়ার লিংক</b>\n\n"
            "লিংক দেখতে, মুছতে বা নতুন লিংক বানাতে "
            "নিচের বাটন ব্যবহার করুন।",
            parse_mode=ParseMode.HTML,
            reply_markup=share_link_keyboard())
        return
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

    # Forwarded group/channel message -> show its chat title + ID
    # (lets the admin verify which chat a forward came from).
    origin = getattr(msg, "forward_origin", None)
    origin_chat = None
    if origin is not None:
        origin_chat = (getattr(origin, "chat", None)
                       or getattr(origin, "sender_chat", None))
    if (origin_chat is not None
            and getattr(origin_chat, "type", "") in
            ("group", "supergroup", "channel")):
        await msg.reply_text(
            f"📣 <b>ফরওয়ার্ড করা চ্যাট</b>\n\n"
            f"নাম: {origin_chat.title}\n"
            f"চ্যাট আইডি: <code>{origin_chat.id}</code>",
            parse_mode=ParseMode.HTML,
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
                    # Auto-translate the reply to the user's
                    # language (Abu, 2026-11-24): the language of
                    # the message being replied to, else the user's
                    # Language setting.
                    target_lang = get_lang(target_id)
                    if replied.text:
                        m2 = re.search(r"মেসেজ👇\s*(.*)",
                                       replied.text, re.S)
                        orig = m2.group(1).strip() if m2 else ""
                        if orig and not orig.startswith("📷"):
                            target_lang = detect_text_lang(orig)
                    send_text, was_tr = await maybe_translate_reply(
                        text, target_lang)
                    # Text-to-Voice toggle decides: voice when ON,
                    # plain text when OFF.
                    await send_admin_reply(context, target_id, send_text)
                else:
                    await context.bot.copy_message(
                        chat_id=target_id,
                        from_chat_id=msg.chat_id,
                        message_id=msg.message_id)
                confirm = "✅ উত্তর পাঠানো হয়েছে।"
                if text and was_tr:
                    confirm += " (ইউজারের ভাষায় অনুবাদ করা হয়েছে)"
                await msg.reply_text(confirm)
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
    # Some buttons also carry a leading emoji (e.g. '🔗 Share
    # link') — strip it so the tap still matches its label.
    plain = strip_button_emoji(from_bold(text))

    # --- Admin side ---
    if is_admin(user.id):
        await handle_admin_message(update, context, msg, text, plain)
        return

    # --- Global force join: non-admin users must be members of
    # the force-join group to use the bot at all (text, voice,
    # photo — everything). The verify context is "start", so
    # after joining they get the welcome.
    gate = await force_join_check(context, user.id)
    if gate == "block":
        await msg.reply_text(
            FJ_PROMPT_TEXT, parse_mode=ParseMode.HTML,
            reply_markup=force_join_markup("start"))
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
    if plain == LBL_HELP:
        await msg.reply_text(help_text(get_lang(user.id)),
                             parse_mode=ParseMode.HTML,
                             reply_markup=main_keyboard())
        return
    if plain == LBL_OWNER:
        await msg.reply_text(owner_text(get_lang(user.id)),
                             parse_mode=ParseMode.HTML,
                             reply_markup=main_keyboard())
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
        ], [
            InlineKeyboardButton(
                "🇨🇳 中文" + (" ✅" if cur == "zh" else ""),
                callback_data="lang:zh", style="primary"),
        ]])
        await msg.reply_text(
            "কোন ভাষায় উত্তর চান? 🙂\nWhich language do you want?\n"
            "您想要哪种语言？",
            reply_markup=markup)
        return

    # 2) Owner question: answer from the intro, do NOT send it on.
    if text and any(w in text.lower() for w in OWNER_WORDS):
        reply = owner_text(get_lang(user.id))
        await msg.reply_text(reply, parse_mode=ParseMode.HTML,
                             reply_markup=main_keyboard())
        log.info("owner intro shown to user %s", user.id)
        return

    # 3) Old keyboard taps (Talk to Admin / AI Chat) from before the
    # AI removal: no such buttons exist any more — every message
    # goes straight to the admin, so just say so and show the new
    # keyboard instead of forwarding the button label as a message.
    if plain in ("Talk to Admin", "AI Chat"):
        await msg.reply_text(
            "✍️ আপনার মেসেজ সরাসরি এডমিনের কাছে যায় — লিখুন, "
            "এডমিন নিজে উত্তর দেবেন।",
            reply_markup=main_keyboard())
        return

    # 4) Everything else goes straight to the admin — text, voice,
    # photo, any media. No AI anywhere.
    await send_to_admin(update, context, msg)
    return


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
        # Auto-translate the reply to the user's language (their
        # Language setting; no original message here to detect from).
        send_text, was_tr = await maybe_translate_reply(
            text, get_lang(target_id))
        # Text-to-Voice toggle decides: voice when ON (text
        # fallback if TTS fails), plain text when OFF.
        await send_admin_reply(context, target_id, send_text)
        confirm = "✅ উত্তর পাঠানো হয়েছে।"
        if was_tr:
            confirm += " (ইউজারের ভাষায় অনুবাদ করা হয়েছে)"
        await update.message.reply_text(confirm)
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
    if not is_admin(user.id):
        # Global force join applies to /owner too.
        gate = await force_join_check(context, user.id)
        if gate == "block":
            await update.message.reply_text(
                FJ_PROMPT_TEXT, parse_mode=ParseMode.HTML,
                reply_markup=force_join_markup("start"))
            return
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


async def mediastats_cmd(update: Update,
                         context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _admin_only(update):
        return
    try:
        total, latest, by_type = media_stats()
    except Exception:
        log.exception("mediastats failed")
        await update.message.reply_text(
            "❌ পরিসংখ্যান দেখা যায়নি, আবার চেষ্টা করুন।",
            reply_markup=admin_keyboard())
        return
    lines = ["📊 <b>মিডিয়া পরিসংখ্যান</b>\n",
             f"📁 মোট ফাইল: {total}",
             f"🆕 সর্বশেষ ID: {latest or '—'}"]
    if by_type:
        lines.append("\nধরন অনুযায়ী:")
        lines.extend(f"• {t}: {c}" for t, c in sorted(by_type.items()))
    await update.message.reply_text(
        "\n".join(lines), parse_mode=ParseMode.HTML,
        reply_markup=admin_keyboard())


async def mediasearch_cmd(update: Update,
                          context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _admin_only(update):
        return
    args = context.args or []
    if not args:
        await update.message.reply_text(
            "ব্যবহার: /mediasearch 00001", reply_markup=admin_keyboard())
        return
    code = args[0].strip()
    if not is_valid_media_code(code):
        await update.message.reply_text(
            "Sorry, this file could not be found.")
        return
    try:
        row = media_get_by_code(code)
    except Exception:
        log.exception("mediasearch lookup failed")
        row = None
    if row is None:
        await update.message.reply_text(
            "Sorry, this file could not be found.")
        return
    await update.message.reply_text(
        f"🔗 File ID: <code>{code}</code>\n"
        f"Share Link:\n{media_share_link(code)}",
        parse_mode=ParseMode.HTML, reply_markup=admin_keyboard())
    if not await send_stored_media(context, update.effective_chat.id,
                                   row):
        await update.message.reply_text("❌ ফাইল পাঠানো যায়নি।")


async def mediadelete_cmd(update: Update,
                          context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _admin_only(update):
        return
    args = context.args or []
    if not args:
        await update.message.reply_text(
            "ব্যবহার: /mediadelete 00001", reply_markup=admin_keyboard())
        return
    code = args[0].strip()
    if not is_valid_media_code(code):
        await update.message.reply_text(
            "Sorry, this file could not be found.")
        return
    status = await delete_media_by_code(context, code)
    if status == "ok":
        await update.message.reply_text(
            f"🗑️ ফাইল <code>{code}</code> মুছে ফেলা হয়েছে।",
            parse_mode=ParseMode.HTML, reply_markup=admin_keyboard())
    elif status == "not_found":
        await update.message.reply_text(
            "Sorry, this file could not be found.")
    else:
        await update.message.reply_text(
            "❌ মুছা যায়নি, আবার চেষ্টা করুন।",
            reply_markup=admin_keyboard())


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
    app = Application.builder().token(TOKEN).request(request).post_init(
        force_join_post_init).build()
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
    app.add_handler(CommandHandler("mediastats", mediastats_cmd))
    app.add_handler(CommandHandler("mediasearch", mediasearch_cmd))
    app.add_handler(CommandHandler("mediadelete", mediadelete_cmd))
    app.add_handler(CallbackQueryHandler(lang_callback,
                                         pattern=r"^lang:(bn|en|zh)$"))
    app.add_handler(CallbackQueryHandler(translate_callback,
                                         pattern=r"^tr:"))
    app.add_handler(CallbackQueryHandler(force_join_callback,
                                         pattern=r"^fjcheck:"))
    app.add_handler(CallbackQueryHandler(force_remove_callback,
                                         pattern=r"^fjremove:"))
    app.add_handler(ChatMemberHandler(
        my_chat_member_update,
        chat_member_types=ChatMemberHandler.MY_CHAT_MEMBER))
    app.add_handler(ChatMemberHandler(
        force_join_member_update,
        chat_member_types=ChatMemberHandler.CHAT_MEMBER))
    app.add_handler(ChatJoinRequestHandler(force_join_request_update))
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
    log.info("starting contact bot v6 (polling), admin=%s, "
             "openrouter=%s, token=%s",
             ADMIN_ID, "on" if OPENROUTER_AVAILABLE else "off",
             TOKEN_SOURCE)
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
