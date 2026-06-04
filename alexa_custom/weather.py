from __future__ import annotations

import logging
from typing import Tuple

import httpx

logger = logging.getLogger(__name__)

_KNOWN_CITIES: dict[str, Tuple[float, float, str]] = {
    "verona": (45.4384, 10.9916, "Verona"),
    "san giovanni lupatoto": (45.3833, 11.0333, "San Giovanni Lupatoto"),
    "montorio veronese": (45.4500, 11.0000, "Montorio Veronese"),
}

_WEATHER_CODES: dict[int, str] = {
    0: "sereno",
    1: "prevalentemente sereno",
    2: "parzialmente nuvoloso",
    3: "coperto",
    45: "nebbioso",
    48: "nebbia con depositi di brina",
    51: "pioggerella leggera",
    53: "pioggerella moderata",
    55: "pioggerella intensa",
    56: "pioggia gelata leggera",
    57: "pioggia gelata intensa",
    61: "pioggia leggera",
    63: "pioggia moderata",
    65: "pioggia intensa",
    66: "pioggia gelata leggera",
    67: "pioggia gelata intensa",
    71: "neve leggera",
    73: "neve moderata",
    75: "neve intensa",
    77: "granelli di neve",
    80: "rovesci leggeri",
    81: "rovesci moderati",
    82: "rovesci violenti",
    85: "rovesci di neve leggeri",
    86: "rovesci di neve intensi",
    95: "temporale",
    96: "temporale con grandine leggera",
    99: "temporale con grandine intensa",
}

_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_WARN_RAIN_HEAVY = 5
_WARN_SNOW_HEAVY = 1
_WARN_WIND_GUST = 60
_WARN_PRECIP_HEAVY = 15

_WEEKDAY_NAMES = (
    "luned\u00ec", "marted\u00ec", "mercoled\u00ec",
    "gioved\u00ec", "venerd\u00ec", "sabato", "domenica",
)

_MONTH_NAMES = (
    "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
    "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
)

_WEEKDAYS = {
    "lunedi": 0, "martedi": 1, "mercoledi": 2,
    "giovedi": 3, "venerdi": 4, "sabato": 5, "domenica": 6,
}


def today_date_text(today=None) -> str:
    from datetime import date
    if today is None:
        today = date.today()
    wd = _WEEKDAY_NAMES[today.weekday()]
    return f"Oggi \u00e8 {wd} {today.day} {_MONTH_NAMES[today.month - 1]} {today.year}"


def _day_label(offset: int, today=None) -> str:
    from datetime import date, timedelta
    if today is None:
        today = date.today()
    if offset == 0:
        return "oggi"
    if offset == 1:
        return "domani"
    if offset == 2:
        return "dopodomani"
    target = today + timedelta(days=offset)
    return f"{_WEEKDAY_NAMES[target.weekday()]} {target.day} {_MONTH_NAMES[target.month - 1]}"


def _parse_day(day_str: str | None, today=None) -> int:
    import unicodedata
    from datetime import date
    if today is None:
        today = date.today()
    if not day_str:
        return 0
    nfd = unicodedata.normalize("NFD", day_str.lower())
    s = "".join(c for c in nfd if not unicodedata.combining(c))
    s = unicodedata.normalize("NFC", s).strip()
    if s in ("oggi",):
        return 0
    if s == "domani":
        return 1
    if s == "dopodomani":
        return 2
    if s in _WEEKDAYS:
        target = _WEEKDAYS[s]
        current = today.weekday()
        days_ahead = target - current
        if days_ahead <= 0:
            days_ahead += 7
        return days_ahead
    try:
        return min(int(s), 14)
    except ValueError:
        return 0


def weather_code_to_text(code: int) -> str:
    return _WEATHER_CODES.get(code, "condizioni sconosciute")


async def geocode(city: str) -> Tuple[float, float, str]:
    key = city.lower().strip()
    if key in _KNOWN_CITIES:
        return _KNOWN_CITIES[key]

    async with httpx.AsyncClient(timeout=5.0) as client:
        params = {"name": city, "count": 1, "language": "it", "format": "json"}
        resp = await client.get(_GEOCODING_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results")
        if not results:
            raise ValueError(f"Citt\u00e0 non trovata: {city!r}")
        r = results[0]
        lat = r["latitude"]
        lon = r["longitude"]
        name = r.get("name", city)
        admin = r.get("admin1", "")
        return lat, lon, f"{name}, {admin}" if admin else name


async def geolocate_ip() -> Tuple[float, float, str]:
    last_err: Exception | None = None
    for retry in range(2):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    "http://ip-api.com/json/?fields=city,regionName,zip,lat,lon"
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("status") == "fail":
                    raise ValueError("GeoIP lookup failed: " + data.get("message", ""))
                city = data.get("city", "qui")
                region = data.get("regionName", "")
                zip_code = data.get("zip", "")
                parts = [city]
                if region:
                    parts.append(region)
                if zip_code:
                    parts.append(zip_code)
                display = ", ".join(parts)
                return data["lat"], data["lon"], display
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError("GeoIP unavailable") from last_err


_hourly_for_warnings = (
    "rain,snowfall,wind_gusts_10m,weather_code,precipitation"
)


async def get_forecast(lat: float, lon: float, days: int = 1) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_sum",
        "hourly": _hourly_for_warnings,
        "timezone": "auto",
        "forecast_days": days,
    }
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(_FORECAST_URL, params=params)
        resp.raise_for_status()
        return resp.json()


def format_forecast(city_name: str, forecast: dict, day_offset: int = 0) -> str:
    from datetime import date

    daily = forecast.get("daily", {})
    if not daily.get("time"):
        return "Non ho trovato dati meteo per questa localit\u00e0"

    idx = min(day_offset, len(daily["time"]) - 1)
    temp_max = daily["temperature_2m_max"][idx]
    temp_min = daily["temperature_2m_min"][idx]
    weather_code = daily["weather_code"][idx]
    precip = daily["precipitation_sum"][idx]

    day_label = _day_label(day_offset, date.today())
    weather_text = weather_code_to_text(weather_code)

    parts = [f"A {city_name} {day_label}"]
    if temp_min != temp_max:
        parts.append(f"da {temp_min:.0f} a {temp_max:.0f} gradi")
    else:
        parts.append(f"{temp_max:.0f} gradi")
    parts.append(weather_text)

    if precip and precip > 0:
        parts.append(f"con {precip:.1f} millimetri di pioggia")

    return ", ".join(parts)


def _time_period(hour: int) -> str:
    if 6 <= hour <= 11:
        return "in mattinata"
    if 12 <= hour <= 13:
        return "intorno a mezzogiorno"
    if 14 <= hour <= 17:
        return "nel pomeriggio"
    if 18 <= hour <= 20:
        return "in serata"
    if 21 <= hour <= 23:
        return "in tarda serata"
    return "durante la notte"


def _group_hours(hours: list[int]) -> str:
    seen: list[str] = []
    for h in sorted(hours):
        label = _time_period(h)
        if not seen or seen[-1] != label:
            seen.append(label)
    if len(seen) == 1:
        return seen[0]
    return ", ".join(seen[:-1]) + " e " + seen[-1]


def check_warnings(forecast: dict, day_offset: int = 0) -> tuple[str, str]:
    """Return (alert_text, info_text) — empty strings if nothing to report."""
    hourly = forecast.get("hourly")
    if not hourly or not hourly.get("time"):
        return "", ""

    times = hourly["time"]
    start = day_offset * 24
    end = min(start + 24, len(times))
    if start >= len(times):
        return "", ""

    day_times = times[start:end]
    rain = (hourly.get("rain", []) or [])[start:end]
    snowfall = (hourly.get("snowfall", []) or [])[start:end]
    gusts = (hourly.get("wind_gusts_10m", []) or [])[start:end]
    codes = (hourly.get("weather_code", []) or [])[start:end]
    precip = (hourly.get("precipitation", []) or [])[start:end]

    rain_any_hours: list[int] = []
    rain_heavy_hours: list[int] = []
    snow_hours: list[int] = []
    gust_hours: list[int] = []
    storm_hours: list[int] = []
    flood_hours: list[int] = []

    for i, t in enumerate(day_times):
        hour = int(t[11:13])
        if i < len(rain) and rain[i] is not None:
            if rain[i] > 0:
                rain_any_hours.append(hour)
            if rain[i] >= _WARN_RAIN_HEAVY:
                rain_heavy_hours.append(hour)
        if i < len(snowfall) and snowfall[i] is not None and snowfall[i] >= _WARN_SNOW_HEAVY:
            snow_hours.append(hour)
        if i < len(gusts) and gusts[i] is not None and gusts[i] >= _WARN_WIND_GUST:
            gust_hours.append(hour)
        if i < len(codes) and codes[i] is not None and codes[i] >= 96:
            storm_hours.append(hour)
        if i < len(precip) and precip[i] is not None and precip[i] >= _WARN_PRECIP_HEAVY:
            flood_hours.append(hour)

    alerts: list[str] = []
    if rain_heavy_hours:
        alerts.append(f"Forti piogge previste {_group_hours(rain_heavy_hours)}")
    if storm_hours:
        alerts.append(f"Temporali con grandine previsti {_group_hours(storm_hours)}")
    if gust_hours:
        alerts.append(f"Forti raffiche di vento previste {_group_hours(gust_hours)}")
    if snow_hours:
        alerts.append(f"Forti nevicate previste {_group_hours(snow_hours)}")
    if flood_hours:
        alerts.append(f"Possibili allagamenti previsti {_group_hours(flood_hours)}")

    infos: list[str] = []
    if rain_any_hours:
        infos.append(f"Pioggia prevista {_group_hours(rain_any_hours)}")

    alert_text = ("Allerta meteo: " + ". ".join(alerts) + ".") if alerts else ""
    info_text = ". ".join(infos) if infos else ""
    return alert_text, info_text
