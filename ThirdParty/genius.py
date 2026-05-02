"""
Genius API client.

Genius API (https://docs.genius.com/) returns song *metadata*

Usage:
    client = Genius(access_token="GENIUS_ACCESS_TOKEN")
    lyrics = client.lyrics("Bohemian Rhapsody", "Queen")
    print(lyrics)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup

# can use another lyric engine here as well, genius has a free API
API_BASE = "https://api.genius.com"


class GeniusError(Exception):
    """Raised for any Genius API or scraping failure."""


@dataclass
class Track:
    id: int
    title: str
    artist: str
    url: str


class Genius:
    def __init__(self, access_token: str, timeout: int = 10):
        if not access_token:
            raise ValueError("A Genius API access token is required.")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {access_token}",
            "User-Agent": "minimal-genius-client/1.0",
        })

    # Public API calls
    def search(self, track: str, artist: str) -> Optional[Track]:
        """Search Genius and return the best-matching Track, or None."""
        query = f"{track} {artist}".strip()
        resp = self._session.get(
            f"{API_BASE}/search",
            params={"q": query},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        hits = resp.json().get("response", {}).get("hits", [])

        artist_lc = artist.lower()
        # Hits where the primary artist matches the requested artist.
        for hit in hits:
            result = hit.get("result", {})
            primary = result.get("primary_artist", {}).get("name", "").lower()
            if artist_lc in primary or primary in artist_lc:
                return self._to_track(result)

        # Fallback: first hit, if any.
        if hits:
            return self._to_track(hits[0]["result"])
        return None

    def lyrics(self, track: str, artist: str) -> str:
        """Return lyrics for the given track + artist."""
        track = self.search(track, artist)
        if track is None:
            raise GeniusError(f"No track found for '{track}' by '{artist}'.")
        return self._scrape_lyrics(track.url)

    @staticmethod
    def _to_track(result: dict) -> Track:
        return Track(
            id=result["id"],
            title=result.get("title", ""),
            artist=result.get("primary_artist", {}).get("name", ""),
            url=result["url"],
        )

    def _scrape_lyrics(self, url: str) -> str:
        resp = requests.get(
            url,
            headers={"User-Agent": "minimal-genius-client/1.0"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Parse divs with `data-lyrics-container` attribute is "true".
        containers = soup.select('div[data-lyrics-container="true"]')
        if not containers:
            raise GeniusError(
                "Could not locate lyrics on the page. Genius may have "
                "changed their HTML structure."
            )

        parts = []
        for container in containers:
            for br in container.find_all("br"):
                br.replace_with("\n")
            parts.append(container.get_text())

        text = "\n".join(parts).
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text
