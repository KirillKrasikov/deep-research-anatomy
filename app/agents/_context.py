from datetime import UTC, datetime


def today_iso() -> str:
    """Текущая дата в UTC (ISO), для подстановки в конец системных промптов (prefix caching)."""
    return datetime.now(UTC).date().isoformat()
