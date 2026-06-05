#!/usr/bin/env python3
import asyncio
import logging
import os
import signal
import threading
from typing import Callable

from livekit.api import AccessToken, VideoGrants
import numpy as np
import shutil

from livekit.rtc import (
    AudioFrame,
    AudioSource,
    AudioStream,
    LocalAudioTrack,
    Room,
    TrackKind,
    TrackPublishOptions,
    TrackSource,
)

from alexa_custom._env import require_env
import sounddevice as sd
from alexa_custom.config import ActionsConfig
from alexa_custom.config_manager import ConfigManager
from alexa_custom.mqtt import MQTTClient

from alexa_custom.audio import (
    find_pipewire_device,
    play_call_end,
    play_call_start,
    set_pipewire_defaults,
)

RECONNECT_DELAY = 5  # seconds between reconnect attempts

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def calculate_peak(frame) -> float:
    """Calculate normalized peak level (0.0-1.0) from an AudioFrame."""
    samples = np.frombuffer(frame.data, dtype=np.int16)
    if len(samples) == 0:
        return 0.0
    # Use int32 for absolute to avoid int16 overflow at -32768
    return float(np.max(np.abs(samples.astype(np.int32)))) / 32768.0


def get_token() -> str:
    api_key = require_env("LIVEKIT_API_KEY")
    api_secret = require_env("LIVEKIT_API_SECRET")
    room = require_env("LIVEKIT_ROOM")
    require_env("LIVEKIT_URL")
    return (
        AccessToken(api_key, api_secret)
        .with_identity("headless-participant")
        .with_name("Headless Participant")
        .with_grants(VideoGrants(room_join=True, room=room))
        .to_jwt()
    )


def make_browser_token(identity: str = "browser-user") -> str:
    """Generate a token for a browser participant with a distinct identity."""
    api_key = require_env("LIVEKIT_API_KEY")
    api_secret = require_env("LIVEKIT_API_SECRET")
    room = require_env("LIVEKIT_ROOM")
    return (
        AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(VideoGrants(room_join=True, room=room))
        .to_jwt()
    )


def browser_join_url(identity: str = "browser-user") -> str:
    """Return the meet.livekit.io URL a browser can open to join the same room."""
    import urllib.parse

    token = make_browser_token(identity)
    room_url = require_env("LIVEKIT_URL")
    require_env("LIVEKIT_ROOM")
    params = urllib.parse.urlencode({"liveKitUrl": room_url, "token": token})
    return f"https://meet.livekit.io/custom/?{params}"


class PipeWireInputCapture:
    """Capture audio from the PipeWire default source via parec into a LiveKit AudioSource.

    Replaces MediaDevices.open_input() which uses PortAudio (broken on ALSA->PipeWire shim).
    """

    def __init__(self, sample_rate: int, num_channels: int = 1):
        self.source = AudioSource(sample_rate=sample_rate, num_channels=num_channels)
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._proc: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def start(self):
        parec = shutil.which("parec")
        if not parec:
            parec = shutil.which("pw-record")
            if not parec:
                raise RuntimeError(
                    "Neither parec nor pw-record found -- cannot capture audio"
                )

        is_pw = "pw-record" in parec
        cmd = [
            parec,
            f"--rate={self._sample_rate}",
            f"--channels={self._num_channels}",
            "--format=s16le" if not is_pw else "--format=s16",
        ]
        if not is_pw:
            cmd.append("--latency-msec=1")

        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

        self._task = asyncio.create_task(self._pump())

    async def _pump(self):
        assert self._proc is not None
        assert self._proc.stdout is not None

        frame_bytes = self._sample_rate * self._num_channels * 2 // 50  # ~20ms

        while not self._stop.is_set():
            try:
                raw = await asyncio.wait_for(
                    self._proc.stdout.read(frame_bytes), timeout=1.0
                )
                if not raw:
                    break

                sample_size = 2 * self._num_channels
                raw = raw[: len(raw) // sample_size * sample_size]
                if not raw:
                    continue

                samples_per_channel = len(raw) // sample_size
                frame = AudioFrame(
                    data=raw,
                    sample_rate=self._sample_rate,
                    num_channels=self._num_channels,
                    samples_per_channel=samples_per_channel,
                )
                await self.source.capture_frame(frame)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"PipeWire capture error: {e}")
                break

        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await self._proc.wait()
            except Exception:
                pass

    async def aclose(self):
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await self._proc.wait()
            except Exception:
                pass
        await self.source.aclose()


class PipeWireOutputPlayer:
    """Play remote audio tracks through pacat (PulseAudio stdin player).

    Replaces MediaDevices.open_output() -> OutputPlayer which uses PortAudio.
    Spawns a single pacat (or paplay) subprocess and pipes all incoming audio frames
    to its stdin. Falls back through pacat → paplay.
    """

    def __init__(self, sample_rate: int, num_channels: int = 1):
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._proc: asyncio.subprocess.Process | None = None
        self._stream_tasks: dict[str, asyncio.Task] = {}
        self._started = False
        self._stop = asyncio.Event()

    async def start(self):
        pacat = shutil.which("pacat")
        if not pacat:
            paplay = shutil.which("paplay")
            if not paplay:
                raise RuntimeError(
                    "Neither pacat nor paplay found -- cannot play audio"
                )
            logger.info("PipeWireOutputPlayer: using paplay (fallback)")
            self._proc = await asyncio.create_subprocess_exec(
                paplay,
                "--raw",
                f"--rate={self._sample_rate}",
                f"--channels={self._num_channels}",
                "--format=s16le",
                stdin=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        else:
            logger.info("PipeWireOutputPlayer: using pacat")
            self._proc = await asyncio.create_subprocess_exec(
                pacat,
                "--raw",
                f"--rate={self._sample_rate}",
                f"--channels={self._num_channels}",
                "--format=s16le",
                "--latency-msec=10",
                stdin=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        self._started = True

    async def add_track(self, track):
        if not self._started:
            return
        stream = AudioStream(
            track,
            sample_rate=self._sample_rate,
            num_channels=self._num_channels,
        )
        task = asyncio.create_task(self._play_stream(track.sid, stream))
        self._stream_tasks[track.sid] = task

    async def remove_track(self, track):
        task = self._stream_tasks.pop(track.sid, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _play_stream(self, sid: str, stream: AudioStream):
        try:
            async for event in stream:
                if self._stop.is_set() or not self._proc:
                    break
                if self._proc.returncode is not None:
                    stderr = (
                        (await self._proc.stderr.read()).decode(errors="replace")
                        if self._proc.stderr
                        else ""
                    )
                    logger.error(
                        "Playback process exited (%d) for track %s: %s",
                        self._proc.returncode,
                        sid,
                        stderr.strip(),
                    )
                    break
                if self._proc.stdin:
                    self._proc.stdin.write(event.frame.data.tobytes())
                    await self._proc.stdin.drain()
        except Exception as e:
            logger.error(f"Playback error for track {sid}: {e}")
        finally:
            await stream.aclose()

    async def aclose(self):
        self._stop.set()
        for tid, task in list(self._stream_tasks.items()):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._stream_tasks.clear()
        if self._proc:
            if self._proc.stdin:
                try:
                    self._proc.stdin.close()
                except Exception:
                    pass
            if self._proc.returncode is None:
                try:
                    self._proc.terminate()
                    await self._proc.wait()
                except Exception:
                    pass


class LiveKitSessionManager:
    """Manages a single LiveKit room session, track publishing, and player state."""

    def __init__(
        self,
        capture: PipeWireInputCapture,
        samplerate: int,
        on_event: Callable[[str, dict], None] | None = None,
    ):
        self.capture = capture
        self.samplerate = samplerate
        self.on_event = on_event
        self.room = Room()
        self.disconnected = asyncio.Event()
        self.call_connected = False
        self.subscribed_tracks: dict[str, AudioStream] = {}
        self.player_tracks: set[str] = set()
        self.volumes = {"mic": 0.0, "spk": 0.0}
        self.tap_tasks: list[asyncio.Task] = []
        self.player = None

        @self.room.on("disconnected")
        def on_disconnected(reason):
            logger.info(f"Room disconnected: {reason}")
            self.emit("disconnected", {"reason": reason})
            self.disconnected.set()

        @self.room.on("track_subscribed")
        def on_track_subscribed(track, publication, participant):
            if track.kind == TrackKind.KIND_AUDIO:
                logger.info(f"Audio track subscribed from {participant.identity}")
                self.emit(
                    "track_subscribed",
                    {"identity": participant.identity, "track_sid": track.sid},
                )
                self.tap_tasks.append(
                    asyncio.create_task(self._tap_remote(participant.identity, track))
                )

                async def _add():
                    try:
                        await self.player.add_track(track)
                        self.player_tracks.add(track.sid)
                        logger.debug(f"Track {track.sid} added to player")
                    except Exception as e:
                        logger.error(f"add_track failed: {e}")

                asyncio.create_task(_add())

        @self.room.on("track_unsubscribed")
        def on_track_unsubscribed(track, publication, participant):
            if track.kind == TrackKind.KIND_AUDIO:
                logger.info(f"Audio track unsubscribed from {participant.identity}")
                self.emit("track_unsubscribed", {"identity": participant.identity})
                self.subscribed_tracks.pop(participant.identity, None)
                if track.sid in self.player_tracks:
                    asyncio.create_task(self.player.remove_track(track))
                    self.player_tracks.discard(track.sid)

        @self.room.on("participant_connected")
        def on_participant_connected(participant):
            logger.info(f"Participant joined: {participant.identity}")
            self.emit("participant_joined", {"identity": participant.identity})

        @self.room.on("participant_disconnected")
        def on_participant_disconnected(participant):
            logger.info(f"Participant left: {participant.identity}")
            self.emit("participant_left", {"identity": participant.identity})

    def emit(self, event: str, data: dict | None = None) -> None:
        if self.on_event:
            self.on_event(event, data or {})

    async def _tap_mic(self, track: LocalAudioTrack, stop_event: asyncio.Event):
        stream = AudioStream(track)
        async for event in stream:
            self.volumes["mic"] = max(self.volumes["mic"], calculate_peak(event.frame))
            if self.disconnected.is_set() or stop_event.is_set():
                break

    async def _tap_remote(self, identity: str, track):
        stream = AudioStream(track)
        self.subscribed_tracks[identity] = stream
        async for event in stream:
            self.volumes["spk"] = max(self.volumes["spk"], calculate_peak(event.frame))
            if identity not in self.subscribed_tracks or self.disconnected.is_set():
                break

    async def _volume_emitter(self, stop_event: asyncio.Event):
        while not self.disconnected.is_set() and not stop_event.is_set():
            await asyncio.sleep(0.1)
            self.emit(
                "volume_update",
                {"mic": self.volumes["mic"], "spk": self.volumes["spk"]},
            )
            self.volumes["mic"] *= 0.6
            self.volumes["spk"] *= 0.6

    async def _empty_room_watchdog(
        self, timeout: float, stop_event: asyncio.Event
    ) -> None:
        import time as _time

        empty_since: float | None = (
            None if self.room.remote_participants else _time.monotonic()
        )
        while not self.disconnected.is_set() and not stop_event.is_set():
            await asyncio.sleep(1.0)
            if self.room.remote_participants:
                empty_since = None
            else:
                now = _time.monotonic()
                if empty_since is None:
                    empty_since = now
                    logger.info(f"Room empty — disconnecting in {timeout:.0f}s")
                elif now - empty_since >= timeout:
                    logger.info("Empty room timeout — disconnecting session")
                    self.emit("empty_room_timeout", {})
                    self.disconnected.set()
                    return

    async def run(self, stop_event: asyncio.Event):
        """Connect to one LiveKit session; return when disconnected or stop_event fires."""
        empty_room_timeout = float(os.environ.get("EMPTY_ROOM_TIMEOUT", "0") or "0")

        self.player = PipeWireOutputPlayer(sample_rate=self.samplerate)
        await self.player.start()
        logger.debug("Session player started")

        self.capture.source.clear_queue()

        try:
            room_url = require_env("LIVEKIT_URL")
            await self.room.connect(room_url, get_token())
            room_name = require_env("LIVEKIT_ROOM")
            logger.info(
                f"Connected to {room_url}/{room_name} as {self.room.local_participant.identity}"
            )
            self.emit(
                "connected",
                {"room": room_name, "identity": self.room.local_participant.identity},
            )
            self.call_connected = True
            asyncio.create_task(asyncio.to_thread(play_call_start))

            for p in self.room.remote_participants.values():
                self.emit("participant_joined", {"identity": p.identity})

            track = LocalAudioTrack.create_audio_track(
                "microphone", self.capture.source
            )
            opts = TrackPublishOptions()
            opts.source = TrackSource.SOURCE_MICROPHONE
            await self.room.local_participant.publish_track(track, opts)
            logger.info("Microphone track published — full duplex active")

            self.tap_tasks.append(asyncio.create_task(self._tap_mic(track, stop_event)))
            self.tap_tasks.append(asyncio.create_task(self._volume_emitter(stop_event)))

            if empty_room_timeout > 0:
                self.tap_tasks.append(
                    asyncio.create_task(
                        self._empty_room_watchdog(empty_room_timeout, stop_event)
                    )
                )

            await asyncio.wait(
                [
                    asyncio.create_task(self.disconnected.wait()),
                    asyncio.create_task(stop_event.wait()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            await self.cleanup()

    async def cleanup(self):
        logger.debug("Cleaning up session tasks and player...")
        for t in self.tap_tasks:
            t.cancel()
        self.tap_tasks.clear()
        self.subscribed_tracks.clear()
        # Signal disconnection early so STT ungates while we finish cleanup.
        # The room "disconnected" event may not fire until room.disconnect() below.
        logger.debug("cleanup: emitting early 'disconnected' to ungate STT")
        self.emit("disconnected", {})
        if self.call_connected:
            try:
                await asyncio.to_thread(play_call_end)
            except Exception:
                pass
        await self.room.disconnect()
        if self.player:
            await self.player.aclose()
        logger.debug("Session cleanup complete")


async def run_session(
    capture: PipeWireInputCapture,
    samplerate: int,
    stop_event: asyncio.Event,
    on_event: Callable[[str, dict], None] | None = None,
):
    """Connect to one LiveKit session; return when disconnected or stop_event fires."""
    manager = LiveKitSessionManager(capture, samplerate, on_event)
    await manager.run(stop_event)


async def _async_main(
    ext_stop_event: asyncio.Event | None = None,
    on_event: Callable[[str, dict], None] | None = None,
    connect_trigger: threading.Event | None = None,
    livekit_connected_flag: threading.Event | None = None,
    actions_config: ActionsConfig | None = None,
    mqtt_client: MQTTClient | None = None,
    stt_ready_event: threading.Event | None = None,
) -> None:
    logger.info(f"Browser join URL:\n  {browser_join_url()}")

    input_spec = os.environ.get("INPUT_DEVICE", "").strip() or None
    output_spec = os.environ.get("OUTPUT_DEVICE", "").strip() or None

    # Route PipeWire to the requested devices, then always talk to LiveKit
    # through the PipeWire virtual device — never open hw: devices directly.
    if input_spec or output_spec:
        await asyncio.to_thread(set_pipewire_defaults, input_spec, output_spec)
        logger.info(
            f"PipeWire routed — input: {input_spec or 'default'}, output: {output_spec or 'default'}"
        )

    pw_device = find_pipewire_device()
    if pw_device is None:
        raise RuntimeError("PipeWire ALSA device not found. Is PipeWire running?")

    in_info = sd.query_devices(pw_device)
    out_info = sd.query_devices(pw_device)

    logger.info(
        f"Input device:  {input_spec or in_info['name']} ({in_info['max_input_channels']} ch)"
    )
    logger.info(
        f"Output device: {output_spec or out_info['name']} ({out_info['max_output_channels']} ch)"
    )

    # Wait for AudioWatcher to configure hardware before executing startup actions
    # (otherwise sounds play through the old default, like HDMI)
    from alexa_custom.audio import check_newpie_ready

    logger.info("Waiting for audio hardware to initialize...")
    for _ in range(15):  # Wait up to 7.5 seconds
        ok, _ = await asyncio.to_thread(check_newpie_ready)
        if ok:
            # Extra settle time for PipeWire/WirePlumber to finalize routing
            await asyncio.sleep(2.0)
            break
        await asyncio.sleep(0.5)

    # Wait for STT backend to finish loading so "Sistema pronto" plays only
    # when the system is actually ready to hear the first wake word.
    if stt_ready_event is not None and not stt_ready_event.is_set():
        logger.info("Waiting for STT backend to initialize...")
        await asyncio.to_thread(stt_ready_event.wait, 60.0)

    # Execute startup actions
    if actions_config and actions_config.on_startup:
        # Prime the audio hardware with a short chime before the first speech
        from alexa_custom.audio import play_tone

        await asyncio.to_thread(play_tone, "startup")
        await asyncio.sleep(0.5)

        logger.info(f"Executing {len(actions_config.on_startup)} startup action(s)")
        from alexa_custom.actions import TelegramClient, _run_action

        # We don't have a connect_fn or connected_flag here in a way that _run_action
        # can use for livekit_join safely during early startup, but we can pass None.
        telegram_client = TelegramClient()
        for action in actions_config.on_startup:
            try:
                await _run_action(
                    action,
                    telegram_client=telegram_client,
                    livekit_connect_fn=None,
                    livekit_connected=False,
                    mqtt_client=mqtt_client,
                )
            except Exception as e:
                logger.error(f"Startup action {action.type} failed: {e}")

    # Use 16kHz for Bluetooth (if we can detect it) or 48kHz for USB/Internal.
    # High sample rates on weak hardware (like Arduino Uno Q) cause mixer timeouts.
    samplerate = 48000
    from alexa_custom.audio import check_newpie_ready

    _, conn_type = await asyncio.to_thread(check_newpie_ready)
    if conn_type == "bluetooth":
        samplerate = 16000
        logger.info("Bluetooth detected — using 16kHz sample rate for session")

    logger.info(f"LiveKit capture via parec (rate={samplerate})")
    capture = PipeWireInputCapture(sample_rate=samplerate, num_channels=1)
    await capture.start()

    # Use the provided stop event (TUI mode) or create one and wire signals.
    stop_event = ext_stop_event or asyncio.Event()
    if ext_stop_event is None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)

    def _wrapped_on_event(event: str, data: dict) -> None:
        if livekit_connected_flag is not None:
            if event == "connected":
                logger.debug("livekit: connected — setting livekit_connected_flag")
                livekit_connected_flag.set()
            elif event == "disconnected":
                logger.debug("livekit: disconnected — clearing livekit_connected_flag")
                livekit_connected_flag.clear()
        if on_event:
            on_event(event, data)

    reconnect_delay = RECONNECT_DELAY
    try:
        while not stop_event.is_set():
            # On-demand mode: wait for STT to signal a connect trigger.
            if connect_trigger is not None:
                logger.info("Waiting for voice trigger to connect to LiveKit…")
                while not stop_event.is_set():
                    if connect_trigger.is_set():
                        connect_trigger.clear()
                        break
                    await asyncio.sleep(0.5)
                if stop_event.is_set():
                    break

            _wrapped_on_event("reconnecting", {})
            connected_this_session = False

            def _on_event_interceptor(event: str, data: dict):
                nonlocal connected_this_session
                if event == "connected":
                    connected_this_session = True
                _wrapped_on_event(event, data)

            try:
                await run_session(
                    capture,
                    samplerate,
                    stop_event,
                    on_event=_on_event_interceptor,
                )
            except Exception as e:
                logger.error(f"Session error: {e}")

            if livekit_connected_flag is not None:
                livekit_connected_flag.clear()

            if stop_event.is_set():
                break

            # In on-demand mode don't auto-reconnect; wait for another trigger.
            if connect_trigger is not None:
                reconnect_delay = RECONNECT_DELAY
                logger.info("LiveKit session ended — waiting for next voice trigger")
                continue

            if connected_this_session:
                reconnect_delay = RECONNECT_DELAY
            else:
                reconnect_delay = min(reconnect_delay * 2, 30)

            logger.info(f"Reconnecting in {reconnect_delay}s...")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=reconnect_delay)
            except asyncio.TimeoutError:
                pass
    finally:
        logger.info("Shutting down LiveKit loop...")
        await capture.aclose()
        logger.info("Done.")


def _mqtt_settings() -> dict:
    return {
        "host": os.environ.get("MQTT_HOST"),
        "port": os.environ.get("MQTT_PORT", "1883"),
        "prefix": os.environ.get("MQTT_TOPIC_PREFIX", "alexa"),
        "node_id": os.environ.get("MQTT_NODE_ID"),
    }


def make_mqtt_reload_callback(
    client_holder: list,  # list[MQTTClient | None] — mutable single-element container
    loop: asyncio.AbstractEventLoop,
):
    """Return a reload callback that reconnects MQTT when broker settings change."""
    prev_settings = _mqtt_settings()

    def _callback(new_config: ActionsConfig) -> None:
        nonlocal prev_settings
        current = _mqtt_settings()
        if current == prev_settings:
            return
        prev_settings = current
        logger.info("MQTT settings changed on reload — reconnecting MQTT client")

        old_client: MQTTClient | None = client_holder[0]
        if old_client is not None:
            asyncio.run_coroutine_threadsafe(old_client.stop(), loop)

        host = current["host"]
        if host:
            new_client = MQTTClient(
                host=host,
                port=int(current["port"]),
                topic_prefix=current["prefix"],
                node_id=current["node_id"],
            )
            asyncio.run_coroutine_threadsafe(new_client.run(), loop)
            client_holder[0] = new_client
        else:
            client_holder[0] = None

    return _callback


def ensure_setup() -> None:
    """Download Vosk model and the default Piper voice if missing."""
    from alexa_custom.setup import download_piper_voice, download_vosk
    from alexa_custom.stt import _MODEL_PATH
    from alexa_custom.tts import PIPER_VOICES_DIR

    if not os.path.isdir(_MODEL_PATH):
        logger.info(
            f"Vosk model not found at {_MODEL_PATH}. Downloading automatically..."
        )
        try:
            download_vosk()
        except Exception as e:
            logger.error(f"Failed to download Vosk model: {e}")

    default_voice = "it_IT-paola-medium"
    voice_onnx = PIPER_VOICES_DIR / f"{default_voice}.onnx"
    if not voice_onnx.is_file():
        logger.info(
            f"Piper voice not found at {voice_onnx}. Downloading automatically..."
        )
        try:
            download_piper_voice(default_voice)
        except Exception as e:
            logger.error(f"Failed to download Piper voice: {e}")


def main() -> None:
    import argparse
    import threading

    from alexa_custom.config import load_config

    config = load_config("config.yaml")
    config_manager = ConfigManager(config)

    ensure_setup()

    from alexa_custom import __version__

    parser = argparse.ArgumentParser(description="alexa-custom LiveKit client")
    parser.add_argument(
        "--version", action="version", version=f"alexa-custom {__version__}"
    )
    parser.add_argument("--web", action="store_true", help="Launch web dashboard")
    parser.add_argument(
        "--web-port", type=int, default=None, help="Web dashboard port (default: 8080)"
    )
    parser.add_argument(
        "--hot-reload", action="store_true", help="Auto-restart on .py file changes"
    )
    args = parser.parse_args()

    if args.hot_reload:
        logger.info("Hot-reload enabled (watching alexa_custom/*.py)")

    if args.web:
        from alexa_custom.web import run_web
        from alexa_custom.config import load_web_config

        input_spec = os.environ.get("INPUT_DEVICE", "").strip() or None
        output_spec = os.environ.get("OUTPUT_DEVICE", "").strip() or None
        room = os.environ.get("LIVEKIT_ROOM", "")

        # Port: CLI flag > config.yaml web.port > default 8080
        web_cfg = load_web_config()
        web_port = args.web_port or int(web_cfg.get("port", 8080))

        connect_trigger: threading.Event | None = None
        livekit_connected_flag: threading.Event | None = None
        stt_params: dict | None = None

        if config is not None:
            from alexa_custom.actions import TelegramClient
            from alexa_custom.tts import init_engine

            connect_trigger = threading.Event()
            livekit_connected_flag = threading.Event()

            init_engine(
                backend_type=config.tts_backend,
                voice=config.tts_voice,
                stt_gated_flag=livekit_connected_flag,
                preroll_ms=config.tts_preroll_ms,
            )

            async def _livekit_connect_fn_web() -> None:
                assert connect_trigger is not None
                connect_trigger.set()

            stt_params = {
                "config": config,
                "stop_event": threading.Event(),
                "telegram_client": TelegramClient(),
                "connect_fn": _livekit_connect_fn_web,
                "connected_flag": livekit_connected_flag,
            }

        async def _run_for_web(
            stop_threading: threading.Event,
            on_event: Callable,
            stop_asyncio: asyncio.Event,
        ) -> None:
            await _async_main(
                ext_stop_event=stop_asyncio,
                on_event=on_event,
                connect_trigger=connect_trigger,
                livekit_connected_flag=livekit_connected_flag,
                actions_config=config,
            )

        run_web(
            run_fn=_run_for_web,
            input_spec=input_spec,
            output_spec=output_spec,
            room=room,
            stt_params=stt_params,
            port=web_port,
            hot_reload=args.hot_reload,
        )

        import time as _time
        import os as _os

        _time.sleep(0.2)
        _os._exit(0)

    else:
        from alexa_custom.audio import AudioWatcher

        input_spec = os.environ.get("INPUT_DEVICE", "").strip() or None
        output_spec = os.environ.get("OUTPUT_DEVICE", "").strip() or None
        output_volume = config.output_volume if config is not None else 0.5

        audio_watcher = AudioWatcher(
            input_spec=input_spec,
            output_spec=output_spec,
            output_volume=output_volume,
        )
        audio_watcher.start()

        if config is not None:
            from alexa_custom.actions import TelegramClient, _run_action
            from alexa_custom.stt import start_stt_thread
            from alexa_custom.tts import init_engine

            connect_trigger = threading.Event()
            livekit_connected_flag = threading.Event()
            telegram_client = TelegramClient()
            stt_stop = threading.Event()

            # Initialize MQTT if configured
            mqtt_client: MQTTClient | None = None
            mqtt_holder: list = [None]
            mqtt_host = os.environ.get("MQTT_HOST")
            if mqtt_host:
                mqtt_port = int(os.environ.get("MQTT_PORT", "1883"))
                mqtt_prefix = os.environ.get("MQTT_TOPIC_PREFIX", "alexa")
                mqtt_node = os.environ.get("MQTT_NODE_ID")
                mqtt_client = MQTTClient(
                    host=mqtt_host,
                    port=mqtt_port,
                    topic_prefix=mqtt_prefix,
                    node_id=mqtt_node,
                )
                mqtt_holder[0] = mqtt_client

            # Initialize TTS with gating
            init_engine(
                backend_type=config.tts_backend,
                voice=config.tts_voice,
                stt_gated_flag=livekit_connected_flag,
                preroll_ms=config.tts_preroll_ms,
            )

            async def _livekit_connect_fn() -> None:
                connect_trigger.set()

            async def _run_main_loop():
                main_tasks = []
                loop = asyncio.get_running_loop()

                # Wire TimerManager to the event loop
                from alexa_custom.timer_manager import manager as timer_manager

                timer_manager.set_loop(loop)

                # Register MQTT reload callback (reconnects if broker settings change)
                mqtt_reload_cb = make_mqtt_reload_callback(mqtt_holder, loop)
                config_manager.register_reload_callback(mqtt_reload_cb)

                # Start config file watcher
                config_manager.start_watcher("config.yaml")

                if args.hot_reload:
                    config_manager.start_source_watcher("alexa_custom")

                if mqtt_client:
                    # Setup callback for incoming MQTT actions
                    async def _on_mqtt_action(action_data: dict):
                        from alexa_custom.config import ActionEntry

                        action = ActionEntry(
                            type=action_data["type"],
                            params=action_data.get("params", {}),
                        )
                        logger.info(f"Executing remote action from MQTT: {action.type}")
                        active_mqtt = mqtt_holder[0]
                        await _run_action(
                            action,
                            telegram_client=telegram_client,
                            livekit_connect_fn=_livekit_connect_fn,
                            livekit_connected=livekit_connected_flag.is_set(),
                            mqtt_client=active_mqtt,
                        )

                    mqtt_client.set_on_command(_on_mqtt_action)
                    main_tasks.append(asyncio.create_task(mqtt_client.run()))

                stt_ready = threading.Event()
                start_stt_thread(
                    config=lambda: config_manager.config,
                    stop_event=stt_stop,
                    telegram_client=telegram_client,
                    livekit_connect_fn=_livekit_connect_fn,
                    livekit_connected_flag=livekit_connected_flag,
                    mqtt_client=mqtt_client,
                    loop=loop,
                    stt_ready_event=stt_ready,
                )
                current = config_manager.config
                logger.info(
                    f"STT started — wake words: {current.wake_words}, "
                    f"{len(current.triggers)} trigger(s) configured"
                )

                main_tasks.append(
                    asyncio.create_task(
                        _async_main(
                            connect_trigger=connect_trigger,
                            livekit_connected_flag=livekit_connected_flag,
                            actions_config=config_manager.config,
                            mqtt_client=mqtt_client,
                            stt_ready_event=stt_ready,
                        )
                    )
                )

                try:
                    await asyncio.gather(*main_tasks)
                finally:
                    config_manager.stop_watcher()

            try:
                asyncio.run(_run_main_loop())
            finally:
                stt_stop.set()
                audio_watcher.stop()
        else:
            try:
                asyncio.run(_async_main(actions_config=config))
            finally:
                audio_watcher.stop()


if __name__ == "__main__":
    main()
