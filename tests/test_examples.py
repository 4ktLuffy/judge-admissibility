"""The README's example runs, and does what the README says it does."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'examples'))


async def test_gate_a_change_promotes_the_better_prompt(capsys) -> None:  # type: ignore[no-untyped-def]
    from gate_a_change import main

    assert await main() is True
    out = capsys.readouterr().out
    assert 'PROMOTE' in out and 'serving: candidate' in out
