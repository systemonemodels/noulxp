"""Tiny fixtures: a word-level tokenizer built in memory, nothing downloaded."""

from __future__ import annotations

import itertools
import string
from pathlib import Path

import pytest

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<mask>"]
WORDS = (
    "choice score noul question the a customer wants refund cancel order track delivery "
    "other billing shipping access login password charged twice urgent low medium high "
    "angry yes no false true statement holds does not hold level help support team "
    "which what how is are was this that it to of and in for on with by from context "
    "options answer proposed fit fits Question Options Answer Context Proposed Does "
    "Which fits".split()
)
LETTERS = list(string.ascii_uppercase) + ["".join(p) for p in itertools.product("A", "ABCDEFGHIJ")]
PUNCT = list("():?.,!-/'\"{}[]")


def build_tokenizer(path: Path) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab: dict[str, int] = {}
    for token in SPECIALS + WORDS + LETTERS + PUNCT + [str(d) for d in range(10)]:
        vocab.setdefault(token, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(SPECIALS)
    tok.save(str(path))
    return path


@pytest.fixture(scope="session")
def tokenizer_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tokenizer(tmp_path_factory.mktemp("tok") / "tokenizer.json")


@pytest.fixture(scope="session")
def tokens(tokenizer_file: Path):  # type: ignore[no-untyped-def]
    from noulxp.tokens import Tokens

    return Tokens(tokenizer_file)
