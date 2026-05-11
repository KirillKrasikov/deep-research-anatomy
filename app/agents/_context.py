from datetime import UTC, datetime


def today_iso() -> str:
    return datetime.now(UTC).date().isoformat()
