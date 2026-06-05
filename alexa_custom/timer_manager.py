from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_ALARM_TIMEOUT_SEC = 60.0
_ALARM_BEEP_INTERVAL = 2.0


@dataclass
class Timer:
    id: int
    duration_sec: float
    status: str
    end_time: float
    created_at: float


_ITALIAN_NUMBERS: dict[str, int] = {
    "un": 1,
    "uno": 1,
    "una": 1,
    "due": 2,
    "tre": 3,
    "quattro": 4,
    "cinque": 5,
    "sei": 6,
    "sette": 7,
    "otto": 8,
    "nove": 9,
    "dieci": 10,
    "undici": 11,
    "dodici": 12,
    "tredici": 13,
    "quattordici": 14,
    "quindici": 15,
    "sedici": 16,
    "diciassette": 17,
    "diciotto": 18,
    "diciannove": 19,
    "venti": 20,
    "trenta": 30,
    "quaranta": 40,
    "cinquanta": 50,
    "sessanta": 60,
    "settanta": 70,
    "ottanta": 80,
    "novanta": 90,
    "cento": 100,
}


def _replace_number_words(text: str) -> str:
    words = text.split()
    result: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w in _ITALIAN_NUMBERS:
            val = _ITALIAN_NUMBERS[w]
            if val == 100 and result and result[-1].isdigit():
                result[-1] = str(int(result[-1]) * 100)
            elif val < 100 and result and result[-1].isdigit():
                result[-1] = str(int(result[-1]) + val)
            else:
                result.append(str(val))
        else:
            result.append(w)
        i += 1
    return " ".join(result)


def parse_duration(text: str) -> float | None:
    if not text:
        return None
    text = text.lower().strip()
    text = _replace_number_words(text)

    total = 0.0

    m = re.search(r"(\d+)\s*(?:ora|ore|h)\b", text)
    if m:
        total += float(m.group(1)) * 3600

    if re.search(r"\bun\s*'?\s*ora\b", text):
        total += 3600

    if re.search(r"\bmezz", text):
        if "ora" in text:
            total += 1800

    m = re.search(r"(\d+)\s*(?:minuto|minuti|min|m)\b", text)
    if m:
        total += float(m.group(1)) * 60

    if re.search(r"\bun\s+minuto\b", text):
        total += 60

    m = re.search(r"(\d+)\s*(?:secondo|secondi|sec|s)\b", text)
    if m:
        total += float(m.group(1))

    if re.search(r"\bun\s+secondo\b", text):
        total += 1

    return total if total > 0 else None


def format_duration(sec: float) -> str:
    total_sec = int(round(sec))
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    seconds = total_sec % 60

    parts = []
    if hours > 0:
        parts.append("1 ora" if hours == 1 else f"{hours} ore")
    if minutes > 0:
        parts.append("1 minuto" if minutes == 1 else f"{minutes} minuti")
    if seconds > 0:
        parts.append("1 secondo" if seconds == 1 else f"{seconds} secondi")

    if not parts:
        return "0 secondi"

    if len(parts) == 1:
        return parts[0]

    return ", ".join(parts[:-1]) + " e " + parts[-1]


class TimerManager:
    def __init__(self):
        self._timers: dict[int, Timer] = {}
        self._next_id = 1
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        self._loop = loop

    def set_timer(self, duration_sec: float) -> int:
        with self._lock:
            timer_id = self._next_id
            self._next_id += 1
            now = time.monotonic()
            timer = Timer(
                id=timer_id,
                duration_sec=duration_sec,
                status="active",
                end_time=now + duration_sec,
                created_at=now,
            )
            self._timers[timer_id] = timer

        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._countdown(timer_id), self._loop)

        return timer_id

    async def _countdown(self, timer_id: int) -> None:
        with self._lock:
            timer = self._timers.get(timer_id)
            if not timer or timer.status != "active":
                return
            remaining = max(0.0, timer.end_time - time.monotonic())

        if remaining > 0:
            await asyncio.sleep(remaining)

        with self._lock:
            timer = self._timers.get(timer_id)
            if not timer or timer.status != "active":
                return
            timer.status = "ringing"

        await self._alarm_loop(timer_id)

    async def _alarm_loop(self, timer_id: int) -> None:
        from alexa_custom.audio import play_beep

        start = time.monotonic()

        while True:
            with self._lock:
                timer = self._timers.get(timer_id)
                if not timer or timer.status != "ringing":
                    return

            if time.monotonic() - start >= _ALARM_TIMEOUT_SEC:
                with self._lock:
                    if timer_id in self._timers:
                        self._timers[timer_id].status = "dismissed"
                logger.info(
                    f"Timer {timer_id}: auto-dismissed after {_ALARM_TIMEOUT_SEC}s"
                )
                return

            await asyncio.to_thread(play_beep, 1046.50, 250)
            await asyncio.sleep(_ALARM_BEEP_INTERVAL)

    def dismiss_ringing(self) -> list[int]:
        dismissed: list[int] = []
        with self._lock:
            for tid, timer in list(self._timers.items()):
                if timer.status == "ringing":
                    timer.status = "dismissed"
                    dismissed.append(tid)
        return dismissed

    def cancel_all(self) -> list[int]:
        cancelled: list[int] = []
        with self._lock:
            for tid, timer in list(self._timers.items()):
                if timer.status in ("active", "ringing"):
                    timer.status = "dismissed"
                    cancelled.append(tid)
        return cancelled

    def has_ringing(self) -> bool:
        with self._lock:
            return any(t.status == "ringing" for t in self._timers.values())

    def has_active(self) -> bool:
        with self._lock:
            return any(t.status == "active" for t in self._timers.values())

    def get_active_timers_text(self) -> str | None:
        now = time.monotonic()
        entries: list[str] = []
        with self._lock:
            for timer in self._timers.values():
                if timer.status == "active":
                    remaining = max(0.0, timer.end_time - now)
                    entries.append(format_duration(remaining))

        if not entries:
            return None

        if len(entries) == 1:
            return f"Manca {entries[0]}"

        parts = ", ".join(f"timer {i + 1}: {e}" for i, e in enumerate(entries))
        return f"Timer attivi: {parts}"


manager = TimerManager()
