"""Генерация data/test_data/raw_data/<label>.jsonl для бенча через локальный POST /v1/chat/completions.

Читает только data/prompt_data/query.jsonl (не перезаписывает его).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

QUERY_REL = Path("data/prompt_data/query.jsonl")
OUTPUT_DIR_REL = Path("data/test_data/raw_data")
DEFAULT_ID_MIN = 51
DEFAULT_ID_MAX = 60
LANGUAGE = "en"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = "compound"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _decode_query_line(path: Path, lineno: int, stripped: str) -> dict[str, Any]:
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as e:
        msg = f"{path}:{lineno}: невалидный JSON"
        raise ValueError(msg) from e


def _store_if_benchmark_row(
    path: Path,
    lineno: int,
    obj: dict[str, Any],
    rows: dict[int, str],
    id_bounds: tuple[int, int],
) -> None:
    id_min, id_max = id_bounds
    if obj.get("language") != LANGUAGE:
        return

    try:
        qid = int(obj["id"])
    except (KeyError, TypeError, ValueError) as e:
        msg = f"{path}:{lineno}: ожидается числовой id"
        raise ValueError(msg) from e

    if qid < id_min or qid > id_max:
        return

    prompt = obj.get("prompt")

    if not isinstance(prompt, str):
        msg = f"{path}:{lineno}: поле prompt должно быть строкой (id={qid})"
        raise TypeError(msg)

    if qid in rows:
        msg = f"{path}: дубликат id={qid}"
        raise ValueError(msg)

    rows[qid] = prompt


def load_queries(path: Path, id_min: int, id_max: int) -> dict[int, str]:
    if not path.is_file():
        msg = f"Нет файла с запросами: {path}"
        raise FileNotFoundError(msg)

    rows: dict[int, str] = {}

    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            obj = _decode_query_line(path, lineno, stripped)
            _store_if_benchmark_row(path, lineno, obj, rows, (id_min, id_max))

    required = set(range(id_min, id_max + 1))
    missing = required - set(rows)

    if missing:
        msg = f"В {path} не хватает записей language={LANGUAGE!r} с id: {sorted(missing)}"
        raise ValueError(msg)

    return rows


def scan_existing_output(path: Path, canonical_prompts: dict[int, str]) -> set[int]:
    if not path.exists():
        return set()

    lines = path.read_text(encoding="utf-8").splitlines()
    done: set[int] = set()

    for i, line in enumerate(lines):
        if not line.strip():
            continue

        try:
            obj: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break

            msg = f"{path}:{i + 1}: битая строка JSON в середине файла"
            raise ValueError(msg) from None

        try:
            qid = int(obj["id"])
        except (KeyError, TypeError, ValueError) as e:
            msg = f"{path}:{i + 1}: ожидается числовой id в строке выхода"
            raise ValueError(msg) from e

        if qid in done:
            msg = f"{path}: повтор id={qid}"
            raise ValueError(msg)

        if qid not in canonical_prompts:
            msg = f"{path}: неизвестный id={qid}"
            raise ValueError(msg)

        if obj.get("prompt") != canonical_prompts[qid]:
            msg = f"{path}: prompt для id={qid} не совпадает с query.jsonl"
            raise ValueError(msg)

        done.add(qid)

    return done


def _article_from_completion_payload(data: dict[str, Any]) -> str:
    choices = data.get("choices")

    if not isinstance(choices, list) or not choices:
        msg = "В ответе API нет choices[0]"
        raise ValueError(msg)

    message = choices[0].get("message")

    if not isinstance(message, dict):
        msg = "В ответе API нет choices[0].message"
        raise TypeError(msg)

    content = message.get("content")

    if not isinstance(content, str):
        msg = "choices[0].message.content должен быть строкой"
        raise TypeError(msg)

    return content


def _sse_payload_from_line(line: str) -> str | None:
    stripped = line.strip()

    if not stripped or stripped.startswith(":"):
        return None

    if not stripped.startswith("data:"):
        return None

    return stripped.removeprefix("data:").lstrip()


def _iter_response_lines(resp: Any) -> Iterator[str]:
    buf = ""

    while True:
        chunk = resp.read(8192)

        if not chunk:
            if buf:
                yield buf.rstrip("\r")

            break

        buf += chunk.decode("utf-8")

        while "\n" in buf:
            line, _, buf = buf.partition("\n")
            yield line.rstrip("\r")


def _read_sse_chat_completion(resp: Any) -> dict[str, Any]:
    last_completion: dict[str, Any] | None = None

    for raw_line in _iter_response_lines(resp):
        payload_str = _sse_payload_from_line(raw_line)

        if payload_str is None:
            continue

        if payload_str == "[DONE]":
            break

        try:
            obj: Any = json.loads(payload_str)
        except json.JSONDecodeError:
            continue

        if isinstance(obj, dict) and isinstance(obj.get("choices"), list):
            last_completion = obj

    if last_completion is None:
        msg = "В потоке SSE не найден финальный JSON с choices"
        raise ValueError(msg)

    return last_completion


def chat_completion_article(
    base_url: str,
    model: str,
    prompt: str,
    task_id: int,
    *,
    use_stream: bool,
) -> str:
    parsed_base = urlparse(base_url)

    if parsed_base.scheme not in {"http", "https"}:
        msg = "--base-url должен начинаться с http:// или https://"
        raise ValueError(msg)

    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": use_stream,
            "user": f"benchmark-generate-{task_id}",
        },
        ensure_ascii=False,
    ).encode("utf-8")

    req = Request(  # noqa: S310 — URL собран из проверенного http(s) base_url
        url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )

    try:
        with urlopen(req, timeout=None) as resp:  # noqa: S310
            if use_stream:
                data = _read_sse_chat_completion(resp)
            else:
                raw = resp.read().decode("utf-8")

                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as e:
                    msg = "Ответ API не JSON"
                    raise ValueError(msg) from e

                if not isinstance(data, dict):
                    msg = "Корень ответа API должен быть объектом JSON"
                    raise TypeError(msg)

    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        msg = f"HTTP {e.code} {e.reason}: {detail}"
        raise RuntimeError(msg) from e
    except URLError as e:
        msg = f"Ошибка соединения с {url!r}: {e.reason!s}"
        raise RuntimeError(msg) from e

    return _article_from_completion_payload(data)


def append_record(path: Path, qid: int, prompt: str, article: str) -> None:
    record = {"id": qid, "prompt": prompt, "article": article}
    line = json.dumps(record, ensure_ascii=False) + "\n"

    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Генерация jsonl для бенча через локальный API.")
    p.add_argument("label", help="Имя файла без пути: data/test_data/raw_data/<label>.jsonl")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"Базовый URL (по умолчанию {DEFAULT_BASE_URL})")
    p.add_argument(
        "--model",
        choices=("react", "compound"),
        default=DEFAULT_MODEL,
        help=f"Поле model в теле запроса (по умолчанию {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--id-min",
        type=int,
        default=DEFAULT_ID_MIN,
        help=f"Минимальный id из query.jsonl (по умолчанию {DEFAULT_ID_MIN})",
    )
    p.add_argument(
        "--id-max",
        type=int,
        default=DEFAULT_ID_MAX,
        help=f"Максимальный id из query.jsonl (по умолчанию {DEFAULT_ID_MAX})",
    )
    p.add_argument(
        "--no-stream",
        action="store_true",
        help="Не использовать stream (один JSON вместо SSE)",
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    root = repo_root()

    if any(c.isspace() for c in args.label):
        msg = "label не должен содержать пробелов"
        raise ValueError(msg)

    query_path = root / QUERY_REL
    out_dir = root / OUTPUT_DIR_REL
    out_path = out_dir / f"{args.label}.jsonl"

    if args.id_min > args.id_max:
        msg = "--id-min не может быть больше --id-max"
        raise ValueError(msg)

    canonical = load_queries(query_path, args.id_min, args.id_max)
    done = scan_existing_output(out_path, canonical)
    out_dir.mkdir(parents=True, exist_ok=True)

    for qid in range(args.id_min, args.id_max + 1):
        if qid in done:
            continue

        prompt = canonical[qid]
        article = chat_completion_article(
            args.base_url,
            args.model,
            prompt,
            qid,
            use_stream=not args.no_stream,
        )
        append_record(out_path, qid, prompt, article)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, TypeError, RuntimeError, OSError) as e:
        sys.stderr.write(f"Ошибка: {e}\n")

        raise SystemExit(1) from e
