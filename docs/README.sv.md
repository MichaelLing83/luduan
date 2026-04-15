# Luduan

[English](README.en.md) | [简体中文](README.zh-CN.md) | [Svenska](README.sv.md) | [Français](README.fr.md)

Luduan är ett tal-till-text-verktyg för macOS med högre noggrannhet än den inbyggda dikteringen i macOS.
Det använder lokal [Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (optimerad för Apple Silicon via MLX) för taligenkänning och en lokal [Ollama](https://ollama.com)-LLM för efterbearbetning.

## Namnets ursprung

Namnet **Luduan** kommer från **甪端**, ett kinesiskt mytologiskt lyckodjur.
På mandarin uttalas **甪端** som **lù duān** (pinyin), ungefär som **"lu-dwan"**.
Det beskrivs ofta som ett enhornat väsen som förknippas med omdöme, rättvisa
och fredlig ordning, och i vissa berättelser kan det förstå mänskligt tal eller
många språk. Därför passar namnet bra för en flerspråkig dikteringsapp som ska
lyssna noggrant och återge tal som tydlig text.

## Funktioner
- **Global snabbtangent** (standard: `Cmd+Shift+Space`) — tryck en gång för att starta, igen för att stoppa
- **Strömmande transkribering** — Whisper bearbetar ljud i block medan du talar
- **LLM-efterbearbetning** — Ollama förbättrar grammatik och interpunktion
- **Valfri appkontext** — kan läsa text före markören för att hjälpa med namn och akronymer
- **Automatisk inklistring** — transkriberad text klistras direkt in i den aktiva appen
- **Menyradsapp** — ligger i macOS-menyraden med statusikoner
- **Helt lokal** — ingen molntjänst används

## Krav
- macOS (Apple Silicon rekommenderas, M1+)
- [Ollama](https://ollama.com) installerad och körs
- Python 3.11+

## Installation

### Bygg som macOS-app (rekommenderas)

Om du bygger en fristående `.app` ger du behörigheter till **Luduan** specifikt, inte till Terminal.

```bash
make app          # bygger dist/Luduan.app
make install      # kopierar till /Applications/Luduan.app
```

Efter installation:
1. Öppna **Systeminställningar → Integritet och säkerhet → Input Monitoring**
2. Lägg till **Luduan** och aktivera det (krävs för den globala snabbtangenten)
3. Starta Luduan från Spotlight eller genom att dubbelklicka i `/Applications`

### Kör direkt från terminalen (utvecklingsläge)

```bash
make dev          # skapar .venv
make run          # kör utan att bygga appen
```

> Obs: om du kör från terminalen måste du ge **Terminal** — inte Luduan — rättighet för Input Monitoring.

### Snabbtangent
- **Cmd+Shift+Space** — starta/stoppa inspelning
- Menyradsikonen visar aktuellt läge: inaktiv 🎙 / inspelning 🔴 / bearbetar ⏳
- I menyraden kan du också välja **språk**, slå på/av **Use App Context** och köra **Prepare Offline Use**

## Konfiguration

Konfigurationsfil: `~/.config/luduan/config.toml`

```toml
[hotkey]
keys = ["cmd", "shift", "space"]

[whisper]
model = "base"
language = ""

[ollama]
host = "http://localhost:11434"
model = "gemma3:4b"
enabled = true

[audio]
sample_rate = 16000
channels = 1

[context]
enabled = false
max_chars = 500
```

## Förbered offlineanvändning

Du kan ladda ner allt Luduan behöver i förväg via menyradsappen:

**Luduan-menyradsikon → Prepare Offline Use**

Detta gör följande:
- cachar den konfigurerade Whisper-modellen lokalt
- ser till att den konfigurerade Ollama-modellen finns lokalt

Förlopp och resultat visas i menyn som en statusrad med **Offline:**, och ett systemmeddelande visas när det är klart.

Efter det kan Luduan användas offline så länge modellerna finns kvar lokalt och Ollama redan är installerat på din Mac.
