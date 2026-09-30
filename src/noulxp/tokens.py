"""Tokenisation with the `tokenizers` library, from the package's tokenizer.json alone.

Every piece is encoded without special tokens: templates place special tokens
themselves, by id.
"""

from __future__ import annotations

from pathlib import Path

from noulxp.errors import PackageError


class Tokens:
    def __init__(self, path: Path) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - a hard dependency
            raise PackageError("the `tokenizers` package is required") from exc
        self.path = Path(path)
        self.backend = Tokenizer.from_file(str(self.path))
        self.backend.no_truncation()
        self.backend.no_padding()

    def encode(self, text: str) -> list[int]:
        ids: list[int] = self.backend.encode(text, add_special_tokens=False).ids
        return ids

    def id(self, token: str) -> int:
        """The id of a token string that must be a single token in the vocabulary."""
        found = self.backend.token_to_id(token)
        if found is None:
            raise PackageError(f"the tokenizer has no token {token!r}")
        return int(found)

    def single(self, text: str) -> int | None:
        """The id `text` encodes to when it is exactly one token, else None."""
        ids = self.encode(text)
        return ids[0] if len(ids) == 1 else None
