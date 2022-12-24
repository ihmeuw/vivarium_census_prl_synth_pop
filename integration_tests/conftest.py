from typing import Dict

import pandas as pd
import pytest
from vivarium import InteractiveContext

from vivarium_census_prl_synth_pop.constants import paths


@pytest.fixture(scope="session")
def sim() -> InteractiveContext:
    """Initialize a simulation for use in tests"""
    simulation = InteractiveContext(paths.MODEL_SPEC_DIR / "model_spec.yaml", setup=False)
    simulation.configuration.population.population_size = 20_000
    simulation.setup()
    return simulation


TIME_STEPS_TO_TEST = [0, 1, 10]


@pytest.fixture(scope="session")
def tracked_live_populations(sim) -> Dict[int, pd.DataFrame]:
    previous_step_number = 0
    population_states = {}
    for step in TIME_STEPS_TO_TEST:
        sim.take_steps(step - previous_step_number)

        pop = sim.get_population()
        population_states[step] = pop[pop["alive"] == "alive"]

        previous_step_number = step
    return population_states
