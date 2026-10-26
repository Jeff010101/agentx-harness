# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for FORK-pin refcounting on ``UserSession`` / ``UserSessionManager``.

A DAG-FORK parent session must remain resident in the worker's session
cache while any of its FORK children still need to dispatch credits
against the parent's history. Refcount semantics:

- ``fork_refcount`` defaults to 0 on a freshly-created session.
- ``pin_for_fork_child`` increments the count (one bump per child seed).
- ``release_fork_child`` decrements, floored at 0 (no negatives).
- ``evict_if_unpinned`` is a no-op while the count is > 0; it removes
  the session from the cache only when the count has reached 0.

The worker call sites that drive this refcount (bump on child seed,
decrement on child join) land in the orchestrator wiring task (P2T26);
this test exercises the storage half only.
"""

import pytest

from aiperf.common.enums import ConversationBranchMode, CreditPhase
from aiperf.common.models.branch import ConversationBranchInfo
from aiperf.common.models.dataset_models import Conversation, Turn
from aiperf.credit.structs import Credit, CreditContext
from aiperf.workers.session_manager import UserSessionManager
from aiperf.workers.worker import Worker


@pytest.fixture
def session_manager() -> UserSessionManager:
    return UserSessionManager()


def _conv_with_n_forks(n: int, session_id: str = "root") -> Conversation:
    branches = [
        ConversationBranchInfo(
            branch_id=f"{session_id}:{i}",
            mode=ConversationBranchMode.FORK,
            child_conversation_ids=[f"child-{i}"],
        )
        for i in range(n)
    ]
    return Conversation(
        session_id=session_id,
        turns=[Turn(branch_ids=[f"{session_id}:{i}" for i in range(n)])],
        branches=branches,
    )


class TestForkRefcount:
    def test_default_refcount_zero(self, session_manager: UserSessionManager) -> None:
        session = session_manager.create_and_store(
            x_correlation_id="corr-zero",
            conversation=_conv_with_n_forks(2),
            num_turns=1,
        )
        assert session.fork_refcount == 0

    def test_increment_per_child_seed(
        self, session_manager: UserSessionManager
    ) -> None:
        session = session_manager.create_and_store(
            x_correlation_id="corr-inc",
            conversation=_conv_with_n_forks(3),
            num_turns=1,
        )
        for _ in range(3):
            session_manager.pin_for_fork_child("corr-inc")
        assert session.fork_refcount == 3

    def test_decrement_on_child_join(self, session_manager: UserSessionManager) -> None:
        session = session_manager.create_and_store(
            x_correlation_id="corr-dec",
            conversation=_conv_with_n_forks(2),
            num_turns=1,
        )
        session_manager.pin_for_fork_child("corr-dec")
        session_manager.pin_for_fork_child("corr-dec")
        session_manager.release_fork_child("corr-dec")
        assert session.fork_refcount == 1

    def test_evict_only_when_refcount_zero(
        self, session_manager: UserSessionManager
    ) -> None:
        session_manager.create_and_store(
            x_correlation_id="corr-evict",
            conversation=_conv_with_n_forks(1),
            num_turns=1,
        )
        session_manager.pin_for_fork_child("corr-evict")

        # Pinned: evict_if_unpinned should be a no-op.
        session_manager.evict_if_unpinned("corr-evict")
        assert session_manager.get("corr-evict") is not None

        # Released: evict_if_unpinned should remove it.
        session_manager.release_fork_child("corr-evict")
        session_manager.evict_if_unpinned("corr-evict")
        assert session_manager.get("corr-evict") is None

    def test_release_floor_zero(self, session_manager: UserSessionManager) -> None:
        session = session_manager.create_and_store(
            x_correlation_id="corr-floor",
            conversation=_conv_with_n_forks(1),
            num_turns=1,
        )
        # Already at 0 — release must not go negative.
        session_manager.release_fork_child("corr-floor")
        session_manager.release_fork_child("corr-floor")
        assert session.fork_refcount == 0

    def test_pin_unknown_session_raises(
        self, session_manager: UserSessionManager
    ) -> None:
        with pytest.raises(KeyError):
            session_manager.pin_for_fork_child("does-not-exist")

    def test_release_unknown_session_is_noop(
        self, session_manager: UserSessionManager
    ) -> None:
        # Should not raise — release is best-effort.
        session_manager.release_fork_child("does-not-exist")

    def test_evict_if_unpinned_unknown_session_is_noop(
        self, session_manager: UserSessionManager
    ) -> None:
        session_manager.evict_if_unpinned("does-not-exist")


@pytest.mark.parametrize("first_child", ["child-a", "child-b"])
@pytest.mark.parametrize("one_branch", [False, True])
def test_completed_sibling_does_not_evict_parent_before_other_child_seeds(
    session_manager: UserSessionManager, first_child: str, one_branch: bool
) -> None:
    """Force the CI interleaving without relying on worker or HTTP timing."""
    conversation = _conv_with_n_forks(2)
    if one_branch:
        conversation.branches = [
            ConversationBranchInfo(
                branch_id="root:0",
                mode=ConversationBranchMode.FORK,
                child_conversation_ids=["child-a", "child-b"],
            )
        ]
        conversation.turns[0].branch_ids = ["root:0"]
    conversation.turns[0].raw_messages = [
        {"role": "system", "content": "root system prompt"},
        {"role": "user", "content": "root user prompt"},
    ]
    parent = session_manager.create_and_store("parent", conversation, num_turns=1)
    parent.advance_turn(0)
    parent.store_response(
        Turn(raw_messages=[{"role": "assistant", "content": "root response"}])
    )
    parent.pending_fork_eviction = True
    session_manager.evict_if_unpinned("parent")
    # The session's branch topology can be dropped on the payload-bytes path.
    parent.conversation.branches = []
    for child_id in (first_child, "child-b" if first_child == "child-a" else "child-a"):
        assert session_manager.get("parent") is parent
        child = session_manager.create_and_store(
            child_id,
            Conversation(
                session_id=child_id,
                turns=[Turn(raw_messages=[{"role": "user", "content": child_id}])],
            ),
            num_turns=1,
            parent_correlation_id="parent",
            branch_mode=ConversationBranchMode.FORK,
        )
        session_manager.pin_for_fork_child("parent")
        session_manager.seed_from_parent(child_id, "parent")
        child.advance_turn(0)
        messages = [
            message for turn in child.turn_list for message in turn.raw_messages
        ]
        assert messages == [
            {"role": "system", "content": "root system prompt"},
            {"role": "user", "content": "root user prompt"},
            {"role": "assistant", "content": "root response"},
            {"role": "user", "content": child_id},
        ]
        session_manager.release_fork_child("parent")
        session_manager.evict(child_id)
    assert session_manager.get("parent") is None


def test_pending_seeds_ignore_spawn_and_unexecuted_turns(
    session_manager: UserSessionManager,
) -> None:
    conversation = Conversation(
        session_id="root",
        turns=[Turn(branch_ids=["fork", "spawn"]), Turn(branch_ids=["later"])],
        branches=[
            ConversationBranchInfo(
                branch_id=branch_id,
                mode=mode,
                child_conversation_ids=children,
            )
            for branch_id, mode, children in [
                ("fork", ConversationBranchMode.FORK, ["child"]),
                ("spawn", ConversationBranchMode.SPAWN, ["spawn-a", "spawn-b"]),
                ("later", ConversationBranchMode.FORK, ["later-a", "later-b"]),
            ]
        ],
    )
    parent = session_manager.create_and_store("parent", conversation, num_turns=1)
    assert parent.fork_children_pending == 1
    parent.pending_fork_eviction = True
    session_manager.pin_for_fork_child("parent")
    session_manager.release_fork_child("parent")
    assert session_manager.get("parent") is None


@pytest.mark.parametrize("first_child_active", [False, True])
def test_worker_final_nonfork_turn_defers_eviction_until_delayed_forks_join(
    session_manager: UserSessionManager, first_child_active: bool
) -> None:
    """Use the real worker terminal handler; no I/O or service startup is needed."""
    conversation = _conv_with_n_forks(2)
    for branch in conversation.branches:
        branch.is_background = True
    conversation.turns.append(Turn())
    parent = session_manager.create_and_store("parent", conversation, num_turns=2)
    parent.advance_turn(0)
    parent.advance_turn(1)
    if first_child_active:
        session_manager.pin_for_fork_child("parent")
    worker = Worker.__new__(Worker)
    worker.session_manager = session_manager
    terminal = Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id="root",
        x_correlation_id="parent",
        turn_index=1,
        num_turns=2,
        issued_at_ns=0,
        has_forks=False,
    )
    worker._release_and_evict_for_terminal(terminal, "parent")
    assert session_manager.get("parent") is parent
    assert parent.pending_fork_eviction is True
    if first_child_active:
        session_manager.release_fork_child("parent")
    else:
        session_manager.pin_for_fork_child("parent")
        session_manager.release_fork_child("parent")
    assert session_manager.get("parent") is parent
    session_manager.pin_for_fork_child("parent")
    session_manager.release_fork_child("parent")
    assert session_manager.get("parent") is None


def test_cancelled_seeded_child_keeps_parent_for_unseeded_sibling(
    session_manager: UserSessionManager,
) -> None:
    parent = session_manager.create_and_store(
        "parent", _conv_with_n_forks(2), num_turns=1
    )
    worker = Worker.__new__(Worker)
    worker.session_manager = session_manager
    parent.pending_fork_eviction = True
    for index, cancelled in enumerate((True, False)):
        child_id = f"child-{index}"
        session_manager.create_and_store(
            child_id, Conversation(turns=[Turn(), Turn()]), num_turns=2
        )
        session_manager.pin_for_fork_child("parent")
        session_manager.seed_from_parent(child_id, "parent")
        credit = Credit(
            id=index,
            phase=CreditPhase.PROFILING,
            conversation_id=child_id,
            x_correlation_id=child_id,
            parent_correlation_id="parent",
            branch_mode=ConversationBranchMode.FORK,
            turn_index=0 if cancelled else 1,
            num_turns=2,
            issued_at_ns=0,
        )
        worker._handle_terminal_disposition(
            credit,
            CreditContext(credit=credit, drop_perf_ns=0, cancelled=cancelled),
            child_id,
        )
        assert session_manager.get(child_id) is None
        assert session_manager.get("parent") is (parent if cancelled else None)
