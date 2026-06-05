from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

WIKI_API = "https://{lang}.wikipedia.org/w/api.php"


_HEADERS = {
    "User-Agent": "alexa-custom/1.0 (assistente vocale; https://github.com/alexa-custom)"
}


async def fetch_wikipedia_context(query: str, lang: str = "it") -> str | None:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 1,
    }
    api_url = WIKI_API.format(lang=lang)
    try:
        async with httpx.AsyncClient(timeout=5.0, headers=_HEADERS) as client:
            resp = await client.get(api_url, params=params)
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("query", {}).get("search", [])
            if not pages:
                return None
            title = pages[0]["title"]

            params2 = {
                "action": "query",
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "titles": title,
                "format": "json",
            }
            resp2 = await client.get(api_url, params=params2)
            resp2.raise_for_status()
            pages2 = resp2.json().get("query", {}).get("pages", {})
            extract = next(iter(pages2.values())).get("extract", "")
            if not extract:
                return None
            return f"Da Wikipedia — {title}:\n{extract[:1000]}"
    except Exception as e:
        logger.debug(f"Wikipedia lookup failed: {e}")
        return None
