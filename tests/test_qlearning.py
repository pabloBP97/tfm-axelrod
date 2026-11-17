import copy

import axelrod as axl
import numpy as np
import pytest

from tfm_axelrod.qlearning import START, QLearner, behaviour, state_key, train

C, D = axl.Action.C, axl.Action.D


def test_state_key():
    assert state_key([], [], 2) == START
    assert state_key([C], [D], 2) == "CD"
    assert state_key([C, D, D], [C, C, D], 2) == "DC,DD"


def test_update_rule_toy():
    q = {START: [0.0, 0.0], "CC": [1.0, 2.0]}
    ag = QLearner(1, q, learning=True, alpha=0.5, gamma=0.9)
    ag.update(START, 0, 3, "CC")          # 0 + 0.5 * (3 + 0.9 * 2 - 0)
    assert q[START] == pytest.approx([2.4, 0.0])
    ag.update("CC", 1, 5, None)           # terminal: 2 + 0.5 * (5 - 2)
    assert q["CC"] == pytest.approx([1.0, 3.5])


def test_update_inside_match():
    """Dos rondas sin exploración contra Cooperator: actualizaciones esperadas."""
    q = {START: [1.0, 0.0], "CC": [1.0, 0.0]}  # prefiere C en ambos estados
    ag = QLearner(1, q, learning=True, alpha=0.5, gamma=0.9, epsilon=0.0)
    axl.Match((ag, axl.Cooperator()), turns=2, seed=0).play()
    # Ronda 2: Q(START,C) = 1 + 0.5 * (3 + 0.9 * 1 - 1) = 2.45
    assert q[START] == pytest.approx([2.45, 0.0])
    ag.finish_episode()
    # Terminal: Q(CC,C) = 1 + 0.5 * (3 - 1) = 2
    assert q["CC"] == pytest.approx([2.0, 0.0])


def test_same_seed_same_result():
    a = train(2, 0.05, 40, seed=7)
    b = train(2, 0.05, 40, seed=7)
    c = train(2, 0.05, 40, seed=8)
    assert a["q"] == b["q"] and a["visits"] == b["visits"]
    np.testing.assert_array_equal(a["curve"], b["curve"])
    assert a["q"] != c["q"]


def test_frozen_agent_valid_and_read_only():
    q = train(2, 0.0, 60, seed=1)["q"]
    snapshot = copy.deepcopy(q)
    agent = QLearner(2, q, learning=False)
    assert axl.Classifiers.obey_axelrod(agent)
    for opp in (axl.TitForTat(), axl.Random(), axl.Defector(), agent.clone()):
        result = axl.Match((agent, opp), turns=50, noise=0.05, seed=3).play()
        assert all(a in (C, D) for a, _ in result)
    assert q == snapshot


def test_frozen_agent_in_tournament():
    q = train(1, 0.0, 30, seed=1)["q"]
    players = [axl.TitForTat(), axl.Defector(), QLearner(1, q, learning=False)]
    res = axl.Tournament(players, turns=20, repetitions=2, seed=1).play(
        progress_bar=False, processes=1)
    assert len(res.ranked_names) == 3


def test_behaviour_tft_vs_defector():
    m = behaviour([("CDDDD", "DDDDD")])
    assert m["coop_rate"] == 0.2
    assert m["defects_first_pct"] == 0.0
    assert m["p_defect_after_opp_defect"] == 1.0
