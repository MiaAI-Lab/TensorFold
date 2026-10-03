"""Rank 0's messages carry a sequence number and a checksum, so rank 1 names a lost, repeated or altered message."""

import time
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from tests.test_cuda_geometry import allocations  # noqa: E402,F401  (fixture: fake triton, so multi imports)


@pytest.fixture
def multi(allocations):  # noqa: F811
    import importlib

    return importlib.import_module("tensorfold.families.glm5_next.cuda.multi")


def test_a_sealed_message_comes_back_whole(multi):
    msg = [6, 3, 1, -1, 2 ** 31 - 1, 0]
    assert multi.unseal(multi.seal(msg, 7), 7) == msg
    assert multi.unseal(multi.seal([], 0), 0) == []


def test_a_long_prompt_digest_stays_in_int32(multi):
    msg = list(range(-(2 ** 31), 2 ** 31, 2 ** 12))           # ~1M values, both ends of int32
    d = multi.digest(msg, 5)
    assert 0 <= d < 2 ** 31 - 1
    assert multi.unseal(multi.seal(msg, 5), 5) == msg


@pytest.mark.parametrize("mangle", [
    lambda m: m[:-3] + m[-2:],                    # a value lost
    lambda m: [m[0] + 1] + m[1:],                 # a value changed
    lambda m: m[:2] + [m[3], m[2]] + m[4:],       # two values swapped
], ids=["lost", "changed", "swapped"])
def test_an_altered_message_is_named(multi, mangle):
    with pytest.raises(RuntimeError, match="checksum differs"):
        multi.unseal(mangle(multi.seal([6, 3, 1, 2, 3], 4)), 4)


def test_a_skipped_or_repeated_message_is_named(multi):
    with pytest.raises(RuntimeError, match="expected rank 0.s message 5, received number 4"):
        multi.unseal(multi.seal([6, 1, 0], 4), 5)
    with pytest.raises(RuntimeError, match="too short"):
        multi.unseal([1], 0)


@pytest.mark.parametrize("value, on", [("", False), ("0", False), ("1", True)])
def test_watchdog_exit_setting(multi, value, on):
    assert multi.watchdog_exits(value) is on
    with pytest.raises(ValueError, match="TF_GLM_MULTI_WATCHDOG_EXIT"):
        multi.watchdog_exits("yes")


def test_both_transports_send_sealed_numbered_messages(multi):
    """``_flush`` seals every message, through ``_share`` or TF_GLM_MULTI_ASYNC's ``_send``, numbered in order."""

    d = object.__new__(multi.MultiDecoder)
    sent, shared = [], []
    d._send = sent.append
    d.g = NS(_share=shared.append, _ring=lambda: None)
    d.idle, d.sent = False, 0
    for async_msg in (False, True, False):
        d.tune, d.outbox = NS(async_msg=async_msg), [6, 1, 0]
        multi.MultiDecoder._flush(d)
        assert d.outbox == []
    assert [multi.unseal(m, n) for m, n in zip(shared, (0, 2))] == [[6, 1, 0], [6, 1, 0]]
    assert multi.unseal(sent[0], 1) == [6, 1, 0]


def test_rank1_applies_unsealed_messages_and_names_a_gap(multi):
    d = object.__new__(multi.MultiDecoder)
    wire = [multi.seal([6, 1, 0], 0), multi.seal([9, 0], 1), multi.seal([6, 1, 1], 3)]
    applied = []
    d.g = NS(_share=lambda values: wire.pop(0))
    d.apply = applied.append
    d.idle, d.watchdog, d.received = False, 0, 0
    d.follow(once=True)
    d.follow(once=True)
    assert applied == [[6, 1, 0], [9, 0]]
    with pytest.raises(RuntimeError, match="expected rank 0.s message 2, received number 3"):
        d.follow(once=True)


def test_health_reports_how_long_the_iteration_has_run(multi):
    d = object.__new__(multi.MultiDecoder)
    d.lanes, d.kept = {}, []
    d.pool = NS(rows=4096, free_rows=lambda: 4096)
    d.iteration_since = None
    assert d.health()["iteration_s"] == 0.0
    d.iteration_since = time.monotonic() - 2.0
    assert d.health()["iteration_s"] >= 2.0
