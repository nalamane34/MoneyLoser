"""Event data sub-package: pluggable event source collection."""

from moneygone.data.events.scraper import EventDataCollector, EventMetadata

__all__ = [
    "EventDataCollector",
    "EventMetadata",
]
