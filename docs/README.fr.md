# Luduan

[English](README.en.md) | [简体中文](README.zh-CN.md) | [Svenska](README.sv.md) | [Français](README.fr.md)

Luduan est un outil de dictée audio-vers-texte pour macOS, conçu pour être plus précis que la dictée intégrée de macOS.
Il utilise [Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) en local (optimisé pour Apple Silicon via MLX) pour la reconnaissance vocale, ainsi qu’un LLM local via [Ollama](https://ollama.com) pour le post-traitement.

## Origine du nom

Le nom **Luduan** vient de **甪端**, une créature auspicieuse de la mythologie chinoise.
En mandarin, **甪端** se prononce **lù duān** (pinyin), avec une approximation en
français de type **« lou-douane »**.
Elle est souvent décrite comme une bête à corne unique, associée au discernement,
à la justice et à un règne paisible, et certaines histoires lui attribuent la
capacité de comprendre la parole humaine ou de nombreuses langues. C’est donc un
nom approprié pour une application de dictée multilingue centrée sur l’écoute
précise et une transcription claire.

## Fonctionnalités
- **Raccourci global** (par défaut : `Cmd+Shift+Space`) — appuyez une fois pour démarrer, puis une seconde fois pour arrêter
- **Transcription en continu** — Whisper traite l’audio par segments pendant que vous parlez
- **Post-traitement par LLM** — Ollama corrige la grammaire et la ponctuation
- **Contexte applicatif optionnel** — peut lire le texte avant le curseur pour mieux gérer les noms et acronymes
- **Collage automatique** — le texte transcrit est collé directement dans l’application active
- **Application de barre de menus** — reste dans la barre de menus macOS avec des icônes d’état
- **Entièrement local** — aucun service cloud

## Prérequis
- macOS (Apple Silicon recommandé, M1+)
- [Ollama](https://ollama.com) installé et en cours d’exécution
- Python 3.11+

## Installation

### Construire comme application macOS (recommandé)

En construisant une application `.app` autonome, vous accordez les autorisations à **Luduan** directement, et non à Terminal.

```bash
make app          # construit dist/Luduan.app
make install      # copie vers /Applications/Luduan.app
make dmg          # construit dist/Luduan.dmg
```

Après l’installation :
1. Ouvrez **Réglages Système → Confidentialité et sécurité → Surveillance de l’entrée**
2. Ajoutez **Luduan** et activez-le (nécessaire pour le raccourci global)
3. Lancez Luduan depuis Spotlight ou en double-cliquant dans `/Applications`

### Exécution directe depuis le terminal (mode développement)

```bash
make dev          # prépare .venv
make run          # exécute sans construire l’application
```

> Remarque : si vous l’exécutez depuis le terminal, vous devez accorder l’autorisation de surveillance de l’entrée à **Terminal**, pas à Luduan.

### Raccourci
- **Cmd+Shift+Space** — démarrer / arrêter l’enregistrement
- L’icône de la barre de menus affiche l’état actuel : inactif 🎙 / enregistrement 🔴 / traitement ⏳
- La barre de menus permet aussi de choisir la **langue**, d’activer **Use App Context** et de lancer **Prepare Offline Use**

## Configuration

Fichier de configuration : `~/.config/luduan/config.toml`

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

## Préparation hors ligne

Vous pouvez pré-télécharger tout ce dont Luduan a besoin depuis l’application de barre de menus :

**Icône Luduan dans la barre de menus → Prepare Offline Use**

Cela permet de :
- mettre en cache localement le modèle Whisper configuré
- vérifier que le modèle Ollama configuré est déjà présent localement

La progression et le résultat apparaissent dans le menu sur une ligne d’état **Offline:**, et une notification système s’affiche une fois l’opération terminée.

Ensuite, Luduan peut être utilisé hors ligne tant que ces modèles restent présents sur le disque et qu’Ollama est déjà installé sur votre Mac.
