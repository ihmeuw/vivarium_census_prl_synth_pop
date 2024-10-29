import os
import warnings
from functools import cache
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import pytest
import scipy.stats
from vivarium import InteractiveContext
from vivarium_public_health import utilities
from vivarium_testing_utils import FuzzyChecker

from vivarium_census_prl_synth_pop.constants import paths


@pytest.fixture(scope="session")
def sim() -> InteractiveContext:
    """Initialize a simulation for use in tests"""
    simulation = InteractiveContext(paths.MODEL_SPEC_DIR / "model_spec.yaml", setup=False)
    simulation.configuration.population.population_size = 250_000
    simulation.setup()
    return simulation


TIME_STEPS_TO_TEST = [0, 1, 10]


@pytest.fixture(scope="session")
def populations(sim) -> List[pd.DataFrame]:
    population_states = []
    for _ in range(max(TIME_STEPS_TO_TEST) + 1):
        pop = sim.get_population(untracked=True).assign(time=sim.current_time)
        pipelines = sim.list_values()
        for pipeline in pipelines:
            p = sim.get_value(pipeline)(pop.index)
            # Metrics is a dict we cannot concat and do not want to
            if isinstance(p, dict):
                continue
            elif isinstance(p, pd.DataFrame):
                pop = pd.concat([pop, p.add_prefix(f"{pipeline}.")], axis=1)
            else:
                # Pipeline is a Series
                pop[pipeline] = p

        population_states.append(pop)
        sim.step()

    return population_states


@pytest.fixture(scope="session")
def tracked_populations(populations) -> List[pd.DataFrame]:
    return [pop[pop["tracked"]] for pop in populations]


@pytest.fixture(scope="session")
def simulants_on_adjacent_timesteps(populations) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    timestep_pairs = []

    for before, after in zip(populations, populations[1:]):
        common_simulants = before.index.intersection(after.index)
        timestep_pairs.append((before.loc[common_simulants], after.loc[common_simulants]))

    return timestep_pairs


@pytest.fixture(scope="session")
def tracked_live_populations(tracked_populations) -> List[pd.DataFrame]:
    return [pop[pop["alive"] == "alive"] for pop in tracked_populations]


@pytest.fixture(scope="session")
def pipeline_columns(sim, populations) -> List[str]:
    pipelines = sim.list_values()
    sample_pop = populations[0]
    pipeline_columns = [
        column
        for column in sample_pop.columns
        for pipeline in pipelines
        if column.startswith(pipeline)
    ]
    return pipeline_columns


@pytest.fixture(scope="session")
def output_directory() -> str:
    v_v_path = Path(os.path.dirname(__file__)) / "v_and_v_output"
    return v_v_path


@pytest.fixture(scope="session")
def fuzzy_checker(output_directory) -> FuzzyChecker:
    checker = FuzzyChecker()

    yield checker

    checker.save_diagnostic_output(output_directory)


# Utilities for working with "multiplicative drifts" -- these are
# pretty specific to this simulation, where we have a lot of fuzzy checks
# that start out precise at the first timestep and gradually become
# fuzzier as the simulation progresses.
# We express these as multiplicative factors applied to the target value
# per unit time.


def multiplicative_drift_to_bound_at_timestep(
    value: float, drift_per_timestep: float, num_timesteps: int
) -> float:
    return value * pow(drift_per_timestep, num_timesteps)


def multiplicative_drifts_to_bounds_at_timestep(
    value: float,
    lower_bound_drift_per_timestep: float,
    upper_bound_drift_per_timestep: float,
    num_timesteps: int,
) -> Tuple[float, float]:
    return (
        multiplicative_drift_to_bound_at_timestep(
            value, lower_bound_drift_per_timestep, num_timesteps
        ),
        multiplicative_drift_to_bound_at_timestep(
            value, upper_bound_drift_per_timestep, num_timesteps
        ),
    )


def multiplicative_drift_to_bound_through_timestep(
    value: float, drift_per_timestep: float, num_timesteps: int
) -> float:
    # Assumption: timesteps have equal sample sizes. This is reasonably accurate.
    return np.mean(
        [
            multiplicative_drift_to_bound_at_timestep(value, drift_per_timestep, x)
            for x in range(num_timesteps)
        ]
    )


def multiplicative_drifts_to_bounds_through_timestep(
    value: float,
    lower_bound_drift_per_timestep: float,
    upper_bound_drift_per_timestep: float,
    num_timesteps: int,
) -> Tuple[float, float]:
    return (
        multiplicative_drift_to_bound_through_timestep(
            value, lower_bound_drift_per_timestep, num_timesteps
        ),
        multiplicative_drift_to_bound_through_timestep(
            value, upper_bound_drift_per_timestep, num_timesteps
        ),
    )


def from_yearly_multiplicative_drift(yearly_drift: float, time_step_days: int) -> float:
    # (Expected) drift per year = drift per timestep ^ (timestep / 1 year)
    # Solve for drift per timestep
    return yearly_drift ** (time_step_days / utilities.DAYS_PER_YEAR)
