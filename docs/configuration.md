# ⚙️ Configuration Guide

Configuration is handled via `.env` for credentials and system settings, and `actions.yaml` for voice command logic.

---

## 1. Environment Variables (`.env`)

Copy the example and fill in your values:
```bash
cp .env.example .env
nano .env
```

### LiveKit Credentials
| Variable | Description |
|----------|-------------|
| `LIVEKIT_URL` | `wss://alexa-custom-u4t05e8q.livekit.cloud` |
| `LIVEKIT_API_KEY` | APIJ5drZtgbbAxQ |
| `LIVEKIT_API_SECRET` | V9lrooIcGPvRWNlaezaEQY9DG1Sm7spLg3dbMeTEuF3 |
| `LIVEKIT_ROOM` | The room name to join |

### Audio Targeting
| Variable | Description |
|----------|-------------|
| `INPUT_DEVICE` | Sounddevice index or name substring (e.g., `NewPie`) |
| `OUTPUT_DEVICE` | Sounddevice index or name substring (e.g., `NewPie`) |

### MQTT (Home Assistant)
| Variable | Description |
|----------|-------------|
| `MQTT_HOST` | Broker IP address |
| `MQTT_PORT` | Broker port (default: `1883`) |
| `MQTT_TOPIC_PREFIX` | Prefix for all topics (default: `alexa`) |
| `MQTT_NODE_ID` | Unique name for this client (default: hostname) |

---

## 2. Voice Actions (`actions.yaml`)

Define your wake words and what happens when you speak a command.

```yaml
wake_words:
  - "galileo"
  - "aiuto"

command_timeout: 3.0
wake_confidence: 0.75

triggers:
  - phrase: "chiama"
    actions:
      - type: livekit_join

  - phrase: "test"
    actions:
      - type: say
        text: "Ricevuto forte e chiaro."
```

### Action Types
- `livekit_join`: Connect to the conference room.
- `telegram`: Send a message (requires `TELEGRAM_BOT_TOKEN`).
- `say`: Speak text via TTS.
- `ask`: Multi-turn dialogue (wait for a specific reply).
- `mqtt_publish`: Send a message to your MQTT broker.
- `shell`: Run a local command.
- `tone`: Play an audio chime (`info`, `success`, `error`).
