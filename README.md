# Mr Support

Telegram admin-contact bot — anyone messages the bot, the admin answers them.
Bot: [@Mr_support0_bot](https://t.me/Mr_support0_bot)

## What it does

- **AI auto-reply** — answers users in Bengali/English (OpenRouter/Groq first, Pollinations free API as fallback).
- **Talk to Admin** — a user taps once and every message goes to the admin until they stop; the admin replies directly to the message.
- **Owner identity** — `/setintro`, `/addinfo`, `/listinfo`, `/delinfo`; anyone can see it with `/owner`. The AI also remembers the last 10 messages of each chat.
- **Voice** — admin replies can be sent as voice messages (Text to Voice toggle, admin only); the AI never sends voice. Voice notes and photos are understood (voice transcription needs an OpenRouter balance top-up).
- **Extras** — urgent-keyword alerts to the admin, 👍/👎 rating under AI replies, spam auto-block, whole-bot language switch (বাংলা / English), surveys/polls, and a nightly summary to the admin.
- **Buttons** — styled reply-keyboard buttons only (no inline buttons), labels in bold Unicode English.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# put your bot token (from @BotFather) and your Telegram user ID in .env
python bot.py
```

`daily_summary.py` sends the nightly summary (run it from cron at 23:55 Asia/Dhaka).

## Notes

- `.env` holds the bot token — **never commit it** (see `.gitignore`).
- `data/` is runtime state (user threads, history, ratings, polls) and is not committed.
- Keep the bot running with a watchdog or process manager; the bot retries transient network errors by itself.
