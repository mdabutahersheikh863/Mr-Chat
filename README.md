# Mr Chat Bot

Telegram admin-contact bot — anyone messages the bot, the admin answers them.
Bot: [@Mr_Chat1bot](https://t.me/Mr_Chat1bot)

Inbox only, no mini app, no auto reactions/views. There is **no AI** in this
bot: every user message (text, voice, photo, any media) goes straight to the
admin, who replies directly to the message.

## What it does

- **Inbox** — user messages reach the admin with name / username / chat ID
  and the message; the admin replies directly to a message (or with
  `/reply <user_id> <text>`). `/broadcast` and `/stats` for the admin.
- **Media sharing ("Share link")** — the admin saves media to a storage
  channel and gets a deep link (`t.me/Mr_Chat1bot?start=00001`); anyone
  opening it receives the file fresh. No forwarding anywhere
  (`protect_content` on). `/mediastats`, `/mediasearch`, `/mediadelete`.
  Click stats: every successful delivery to a user is counted
  (`/mediastats` shows total clicks, unique clickers and top links).
- **Password links** — `/setlinkpass <code> <password>` /
  `/dellinkpass <code>`; opening a protected link asks for the password
  (3 tries, the attempt is never forwarded to the admin).
- **Auto-delete** — `/setautodelete <hours>` (0 = off, max 168); delivered
  files delete themselves after that many hours.
- **Inline mode** — search shared files from any chat
  (`@Mr_Chat1bot <name or File ID>`); members only, password links never
  listed. Needs `/setinline` enabled in BotFather.
- **My files** — `/myfiles` and the 📁 𝐌𝐲 𝐟𝐢𝐥𝐞𝐬 button: the files a user
  opened before.
- **Welcome** — the admin's ✏️ 𝐖𝐞𝐥𝐜𝐨𝐦𝐞 button changes the welcome text
  (`{name}` supported), photo and voice (voice in the admin's cloned
  voice); Reset returns to the default.
- **Multi-admin** — `/addadmin`, `/deladmin`, `/listadmins` (main admin
  only); user messages reach every admin.
- **Force join** — the whole bot is gated on joining a group/channel:
  non-members get a Join button + Verify button. Leaving the group sends
  one private reminder. Join requests are auto-approved, but only when
  they came through the bot's own invite link (the bot creates its own
  link for the group, once, at startup or setup). The default group can
  never be removed. Admin menu: "🔐 𝐅𝐨𝐫𝐜𝐞 𝐂𝐫𝐞𝐚𝐭𝐞" (Force Join /
  Remove / List / Cancel).
- **Translate** — two buttons (🇧🇩 Bn / 🇬🇧 En) under each user message for
  the admin; admin replies auto-translate to the user's language.
- **Text to Voice** — admin toggle: admin text replies also go out as
  voice in the admin's cloned voice.
- **Language** — whole-bot language switch: বাংলা / English / 中文.
- **Extras** — urgent-keyword alerts to the admin, spam auto-block (10th
  message in 60s), owner identity (`/setintro`, `/addinfo`, `/listinfo`,
  `/delinfo`, `/owner`), and a counts-only nightly summary to the admin.
- **Buttons** — styled reply-keyboard buttons (Bot API 9.4 `style`) and
  styled inline buttons; labels in bold Unicode English.

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
The bot retries transient network errors by itself; keep it running with a
watchdog that restarts it when the process is gone.

## Notes

- `.env` holds the bot token — **never commit it** (see `.gitignore`).
- `data/` is runtime state (user threads, media database, force-join config)
  and is not committed.
- The bot needs to be an administrator in the force-join group (with
  `can_invite_users`) for join-request approval and leave reminders, and
  in the storage channel to post and delete media.
