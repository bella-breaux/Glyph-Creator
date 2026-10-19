# Glyph Studio

Design, animate and AI-generate patterns for the **Glyph Matrix** on the Nothing Phone (4a) Pro: 137 LEDs in a 13×13 circle. The AI runs on your own computer through [Ollama](https://ollama.com), so it's free to use as much as you want.

Unofficial fan project. Not affiliated with or endorsed by Nothing Technology.

## How it works

Your computer hosts both the studio and the AI, privately, through [Tailscale](https://tailscale.com):

```
phone (Tailscale on) ──https──▶ your-pc.tailnet.ts.net        → the studio page
                     ──https──▶ your-pc.tailnet.ts.net:8443   → Ollama (AI)
```

Only devices signed into your Tailscale account can open either one. Nothing is exposed to the public internet.

## Set up a computer (one command)

Before you start, sign into Tailscale on your phone. Then in the [Tailscale admin console](https://login.tailscale.com/admin/dns), turn on **MagicDNS** and **HTTPS certificates**.

**Windows**: open PowerShell and paste:

```powershell
irm https://raw.githubusercontent.com/bella-breaux/Glyph-Creator/main/setup/setup-windows.ps1 | iex
```

**Linux / macOS**:

```bash
curl -fsSL https://raw.githubusercontent.com/bella-breaux/Glyph-Creator/main/setup/setup-linux-mac.sh | bash
```

The script:

1. Installs Ollama and Tailscale if they're missing
2. Signs the computer into Tailscale (a browser opens; use the same account as your phone)
3. Installs the latest studio
4. Sets `OLLAMA_HOST` and `OLLAMA_ORIGINS` and restarts Ollama
5. Blocks Ollama's port from other devices on your Wi-Fi
6. Downloads `gemma4:e2b-it-qat` if you have no models
7. Publishes the studio and Ollama on your tailnet with `tailscale serve`
8. Tests both and prints your link

Re-run it any time to update the studio or fix the setup. Settings survive reboots. On macOS, re-run it after a reboot, because macOS forgets the Ollama settings.

**Use it:** on your phone, turn Tailscale on and open the link the script printed, e.g. `https://bellapc.tail383178.ts.net/`. The AI connects by itself. Add the page to your home screen to open it like an app.

**Turn it off:** `tailscale serve reset`

## Studio features

- Draw on the true circular 137-LED layout: pen, eraser, fill, mirror and 4-way symmetry, 0–255 brightness
- Frame timeline with per-frame timing, onion skin, undo/redo
- Built-in animations: ripple, spinner, pulse, rain, breathe, rotate, scrolling text
- Ringtone sync: load a sound, detect beats, fit frames to beats, flash on every beat, or visualize the sound
- Image or GIF to glyph: choose, drop or paste a picture or animated GIF. It removes the background, fits the art onto the circle and gives each region its own shade. Onion skin can then show the source under the LEDs for touch-ups
- AI assistant: new design from a description, animate your drawing, revise an animation, or choreograph to your ringtone
- Export `.glyph.zip` (pattern + ringtone) or `.json` for the phone

## AI assistant

- **Model:** pick any model installed in Ollama. Use **Compare** to run the same request on two models side by side, then keep the better one.
- **Recipe mode (default):** the model writes a short recipe of emoji, text, shapes and effects with motion, and the studio draws it precisely. Small models do this far better than drawing pixels. You can edit the recipe by hand under **Recipe**.
- **Pixels mode:** the model draws every LED itself. Use it only with bigger models.
- **Train your own:** a fine-tuned `glyphgen` model is the best option. See [training/TRAINING.md](training/TRAINING.md). Save good results with **Save as training example** to teach it your style.

Models that fit 8 GB of VRAM:

```bash
ollama pull gemma4:e2b-it-qat   # fast and light; the setup script installs it
ollama pull qwen3.5:9b          # better at following descriptions
```

The studio sends `num_ctx 8192`, `temperature 0.4`, `think: false` and a JSON schema. If a reply is cut off, it keeps the complete parts. If it's slow, run `ollama ps`: anything other than `100% GPU` means the model spilled into RAM.

## Troubleshooting

| Problem | Fix |
|---|---|
| Phone can't open the link | Tailscale is off on the phone, or the computer is asleep or off. |
| "Can't reach Ollama" | Run `tailscale serve status` on the computer. You should see port 443 (site) and 8443 (Ollama). Re-run the setup script if not. |
| "Ollama refused this site" (403) | Ollama started before its settings changed. Quit it fully (tray icon → Quit) and reopen, or re-run the script. |
| Script says HTTPS couldn't publish | Turn on HTTPS certificates in the Tailscale admin console, then re-run. |
| macOS: serve fails on the folder | The App Store version of Tailscale can't serve files. Install the standalone version. |

The studio also works at [bella-breaux.github.io/Glyph-Creator](https://bella-breaux.github.io/Glyph-Creator/). Enter `https://your-pc.tailnet.ts.net:8443` in Connection settings.

## Pattern format (`pattern.json`)

```json
{ "format": "glyphstudio/1", "device": "DEVICE_25111p", "size": 13, "ledCount": 137,
  "name": "Ripple", "loop": true, "totalMs": 980,
  "audio": { "file": "ringtone.mp3", "durationMs": 4200 },
  "frames": [ { "ms": 70, "px": [0, 0, 255, ...169 values, row-major, 0–255] } ] }
```

Cells outside the circle are always `0`.

## Repo layout

| Path | What it is |
|---|---|
| `index.html` | The studio (single file) |
| `setup/setup-windows.ps1` | Windows setup |
| `setup/setup-linux-mac.sh` | Linux / macOS setup |
| `training/` | Dataset generator, pixel-art fetcher and converter, fine-tuning script, evaluator and guide for your own `glyphgen` model |
| `android/` | Companion app that plays patterns on the matrix (coming next) |

The companion app needs `glyph-matrix-sdk-2.0.aar` from Nothing's [GlyphMatrix Developer Kit](https://github.com/Nothing-Developer-Programme/GlyphMatrix-Developer-Kit). Its license doesn't allow redistribution, so it's git-ignored. Download it and place it at `android/app/libs/`.
