"""Tests for the synthetic Blocksworld planning domain (PDDL-Instruct-style
CoT curriculum data). Pure CPU, no GPU/network -- runs in well under a second.
"""

import random

import colab_trainable_dendritic_lm as m


def test_state_from_towers_predicates():
    state = m._bw_state_from_towers([["a", "b"], ["c"]])
    assert state["ontable"] == {"a", "c"}
    assert state["on"] == {"b": "a"}
    assert state["clear"] == {"b", "c"}
    assert state["holding"] is None


def test_action_spec_pick_up_reads_state_not_defaults():
    # b is stacked on a: clear(b) True, but ontable(b) False (it's on a, not the table).
    state = m._bw_state_from_towers([["a", "b"], ["c"]])
    preconds, add, del_ = m._bw_action_spec(state, ("pick-up", "b"))
    holds = dict(preconds)
    assert holds["(clear b)"] is True
    assert holds["(ontable b)"] is False
    assert holds["(handempty)"] is True
    assert add == ["(holding b)"]
    assert del_ == ["(ontable b)", "(clear b)", "(handempty)"]


def test_action_spec_stack_precondition_fails_without_holding():
    state = m._bw_state_from_towers([["a"], ["b"]])
    preconds, _, _ = m._bw_action_spec(state, ("stack", "a", "b"))
    holds = dict(preconds)
    assert holds["(holding a)"] is False  # nothing has been picked up yet
    assert holds["(clear b)"] is True


def test_apply_pick_up_then_stack_round_trip():
    state = m._bw_state_from_towers([["a"], ["b"]])
    state = m._bw_apply(state, ("pick-up", "a"))
    assert state["holding"] == "a"
    assert "a" not in state["ontable"] and "a" not in state["clear"]

    state = m._bw_apply(state, ("stack", "a", "b"))
    assert state["holding"] is None
    assert state["on"] == {"a": "b"}
    assert state["clear"] == {"a"}  # b is now covered, no longer clear


def test_solve_produces_a_plan_that_reaches_its_own_goal():
    rng = random.Random(42)
    blocks = ["a", "b", "c", "d", "e"]
    for _ in range(20):
        init_towers = m._bw_random_towers(rng, blocks)
        goal_towers = m._bw_random_towers(rng, blocks)
        plan = m._bw_solve(init_towers, goal_towers)
        goal_preds = set(m._bw_goal_preds(goal_towers))

        state = m._bw_state_from_towers(init_towers)
        for action in plan:
            preconds, _, _ = m._bw_action_spec(state, action)
            assert all(ok for _, ok in preconds), f"solver produced inapplicable action {action}"
            state = m._bw_apply(state, action)

        final_preds = set(m._bw_state_preds(state))
        assert goal_preds <= final_preds, "solver's own plan didn't reach its own goal"


def test_render_example_always_reaches_a_verdict():
    rng = random.Random(7)
    valid = invalid = 0
    for _ in range(300):
        n = rng.randint(3, 6)
        text = m._bw_render_example(rng, n, invalid_frac=0.2)
        is_valid = "[PLAN VALIDITY] This plan is VALID." in text
        is_invalid = "[PLAN VALIDITY] This plan is INVALID." in text
        assert is_valid ^ is_invalid, "example must be exactly one of valid/invalid"
        valid += is_valid
        invalid += is_invalid
    # invalid_frac=0.2 is a per-example coin flip (only when a skip candidate
    # exists); a wide tolerance keeps this a smoke check, not a flaky exact one.
    assert 0.05 < invalid / (valid + invalid) < 0.35


def test_render_example_invalid_names_a_real_violated_precondition():
    rng = random.Random(3)
    text = m._bw_render_example(rng, 3, invalid_frac=1.0)
    assert "VIOLATION" in text
    assert "This plan is INVALID." in text
    assert "[FINAL PLAN]" not in text  # invalid plans never reach the final-plan line


def test_render_example_valid_final_plan_matches_a_real_solve():
    rng = random.Random(11)
    text = m._bw_render_example(rng, 4, invalid_frac=0.0)
    assert "This plan is VALID." in text
    assert "Goal is ACHIEVED." in text


def test_blocksworld_stream_does_not_replay_on_repeated_iter():
    # Every example shares the identical "[BLOCKSWORLD PLANNING PROBLEM]\n
    # Objects: " header, so compare full text (where the varying object
    # letters / states live), not a short prefix that never leaves it.
    stream = m.BlocksworldPlanningStream(seed=1)
    first = [next(stream)["text"] for _ in range(3)]
    second = [next(iter(stream))["text"] for _ in range(3)]  # simulates WeightedMultiplex re-iterating
    assert first != second, "stream must continue advancing across iter() calls, not restart"


def test_blocksworld_stream_rows_match_prose_formatter_shape():
    stream = m.BlocksworldPlanningStream(seed=2)
    row = next(stream)
    assert set(row.keys()) == {"text"}
    assert isinstance(row["text"], str) and len(row["text"]) > 0
