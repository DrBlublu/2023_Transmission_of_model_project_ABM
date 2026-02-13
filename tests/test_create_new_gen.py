import pandas as pd
from ABM.create_new_gen import create_new_gen


def test_create_new_gen_basic():
    """
    Test basic creation of new generation.
    DataFrame should have n rows and correct columns,
        assert new_gen.shape[0] == n, and
        assert set(new_gen.columns) == expected_columns.
    """
    parents = pd.DataFrame({
        "id": ["p1", "p2"],
        "model_knowledge": [0.5, 0.7],
        "w": [0.2, 0.4],
        "perf": [10, 20]
    })
    gen, n = 1, 2
    new_gen = create_new_gen(parents, gen, n)

    assert new_gen.shape[0] == n
    expected_columns = {
        "id", "gen", "model_knowledge", "w",
        "parent_id", "parent_model_knowledge", "parent_w", "parent_perf"
    }
    assert set(new_gen.columns) == expected_columns


def test_create_new_gen_ids_and_gen_column():
    """
    Test ID and generation column values.
    IDs should follow format g{gen}n{i}, and gen column should equal gen,
        assert new_gen["id"].tolist() == expected_ids, and
        assert all(new_gen["gen"] == gen).
    """
    parents = pd.DataFrame({
        "id": ["p1", "p2", "p3"],
        "model_knowledge": [0.1, 0.2, 0.3],
        "w": [0.4, 0.5, 0.6],
        "perf": [7, 8, 9]
    })
    gen, n = 2, 3
    new_gen = create_new_gen(parents, gen, n)

    expected_ids = ["g2n0", "g2n1", "g2n2"]
    assert new_gen["id"].tolist() == expected_ids
    assert all(new_gen["gen"] == gen)


def test_create_new_gen_parent_values_copied():
    """
    Test that parent values are copied correctly.
    Parent columns in new_gen should equal parent_values columns,
        pd.testing.assert_series_equal(new_gen["parent_id"], parents["id"]).
    """
    parents = pd.DataFrame({
        "id": ["p1", "p2"],
        "model_knowledge": [0.5, 0.7],
        "w": [0.2, 0.4],
        "perf": [10, 20]
    })
    gen, n = 1, 2
    new_gen = create_new_gen(parents, gen, n)

    pd.testing.assert_series_equal(new_gen["parent_id"], parents["id"], check_names=False)
    pd.testing.assert_series_equal(new_gen["parent_model_knowledge"], parents["model_knowledge"], check_names=False)
    pd.testing.assert_series_equal(new_gen["parent_w"], parents["w"], check_names=False)
    pd.testing.assert_series_equal(new_gen["parent_perf"], parents["perf"], check_names=False)
