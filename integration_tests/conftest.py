import os
import warnings
from functools import cache
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
import scipy.stats
from vivarium import InteractiveContext

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


class FuzzyChecker:
    """
    This class manages "fuzzy" checks -- that is, checks of values that are
    subject to stochastic variation.
    It uses statistical hypothesis testing to determine whether the observed
    value in the simulation is extreme enough to reject the null hypothesis that
    the simulation is behaving correctly (according to a supplied verification
    or validation target).

    More detail about the statistics used here can be found at:
    https://vivarium-research.readthedocs.io/en/latest/model_design/vivarium_features/automated_v_and_v/index.html#fuzzy-checking
    """

    def __init__(self) -> None:
        self.comparisons_made = []

    def fuzzy_assert_proportion(
        self,
        name: str,
        boolean_values: pd.Series,
        target_value: Optional[float] = None,
        target_value_lb: Optional[float] = None,
        target_value_ub: Optional[float] = None,
        name_addl: Optional[str] = "",
    ) -> None:
        if target_value is not None:
            target_value_lb = target_value
            target_value_ub = target_value

        if target_value_lb is None or target_value_ub is None:
            raise ValueError(
                f"{name}: Not enough information about the target value supplied"
            )

        assert target_value_ub >= target_value_lb

        numerator = boolean_values.sum()
        denominator = len(boolean_values)
        proportion = boolean_values.mean()

        # TODO: Use a different prior than this Jeffreys prior?
        bug_distribution = scipy.stats.betabinom(a=0.5, b=0.5, n=denominator)

        if target_value_lb == target_value_ub:
            no_bug_distribution = scipy.stats.binom(p=target_value_lb, n=denominator)
        else:
            a, b = self._fit_beta_distribution_to_ui(target_value_lb, target_value_ub)

            no_bug_distribution = scipy.stats.betabinom(a=a, b=b, n=denominator)

        bayes_factor = self._calculate_bayes_factor(
            numerator, bug_distribution, no_bug_distribution
        )

        # TODO: Make this configurable?
        reject_null = bayes_factor > 100
        self.comparisons_made.append(
            {
                "name": name,
                "name_addl": name_addl,
                "proportion": proportion,
                "numerator": numerator,
                "denominator": denominator,
                "target_value_lb": target_value_lb,
                "target_value_ub": target_value_ub,
                "bayes_factor": bayes_factor,
                "reject_null": reject_null,
            }
        )

        if reject_null:
            if boolean_values.mean() < target_value_lb:
                raise AssertionError(
                    f"{name} value {proportion:g} is significantly less than expected, bayes factor = {bayes_factor:g}"
                )
            else:
                raise AssertionError(
                    f"{name} value {proportion:g} is significantly greater than expected, bayes factor = {bayes_factor:g}"
                )

        if (
            target_value_lb > 0
            and self._calculate_bayes_factor(0, bug_distribution, no_bug_distribution) < 100
        ):
            warnings.warn(
                f"Sample size too small to ever find that the simulation's '{name}' value is less than expected."
            )

        if target_value_ub < 1 and (
            self._calculate_bayes_factor(denominator, bug_distribution, no_bug_distribution)
            < 100
        ):
            warnings.warn(
                f"Sample size too small to ever find that the simulation's '{name}' value is greater than expected."
            )

        if 100 > bayes_factor > 0.1:
            warnings.warn(f"Bayes factor for '{name}' is not conclusive.")

    def _calculate_bayes_factor(
        self,
        numerator: int,
        bug_distribution: scipy.stats.rv_discrete,
        no_bug_distribution: scipy.stats.rv_discrete,
    ) -> float:
        # We can be dealing with some _extremely_ unlikely events here, so we have to set numpy to not error
        # if we generate a probability too small to be stored in a floating point number(!), which is known
        # as "underflow"
        with np.errstate(under="ignore"):
            bug_marginal_likelihood = bug_distribution.pmf(numerator)
            no_bug_marginal_likelihood = no_bug_distribution.pmf(numerator)

        try:
            return bug_marginal_likelihood / no_bug_marginal_likelihood
        except (ZeroDivisionError, FloatingPointError):
            return 1_000_000.0

    @cache
    def _fit_beta_distribution_to_ui(self, lb: float, ub: float) -> Tuple[float, float]:
        assert lb > 0 and ub < 1
        # Inspired by https://stats.stackexchange.com/a/112671/
        def objective(x):
            # np.exp ensures they are always positive
            a, b = np.exp(x)
            dist = scipy.stats.beta(a=a, b=b)

            squared_error_lower = self._quantile_squared_error(dist, lb, 0.025)
            squared_error_upper = self._quantile_squared_error(dist, ub, 0.975)

            return squared_error_lower + squared_error_upper

        # It is quite important to start with a reasonable guess.
        ui_midpoint = (lb + ub) / 2
        # TODO: Is this reasonable!?
        for first_guess_concentration in [1_000, 100, 10]:
            optimization_result = scipy.optimize.minimize(
                objective,
                x0=[
                    np.log(ui_midpoint * first_guess_concentration),
                    np.log((1 - ui_midpoint) * first_guess_concentration),
                ],
            )
            # Sometimes it warns that it may not have found a good solution,
            # but the solution is very accurate.
            if optimization_result.success or optimization_result.fun < 1e-05:
                break

        assert optimization_result.success or optimization_result.fun < 1e-05

        result = np.exp(optimization_result.x)
        assert len(result) == 2
        return tuple(result)

    def _quantile_squared_error(
        self, dist: scipy.stats.rv_continuous, value: float, intended_quantile: float
    ) -> float:
        with np.errstate(under="ignore"):
            actual_quantile = dist.cdf(value)

        if 0 < actual_quantile < 1:
            return (
                scipy.special.logit(actual_quantile) - scipy.special.logit(intended_quantile)
            ) ** 2
        else:
            # In this case, we were so far off that the actual quantile can't even be
            # precisely calculated.
            # We return an arbitrarily large penalty to ensure this is never selected as the minimum.
            return 1_000_000.0**2

    def write_output(self) -> None:
        output = pd.DataFrame(self.comparisons_made)
        output.to_csv(
            Path(os.path.dirname(__file__)) / "v_and_v_output/proportion_tests.csv",
            index=False,
        )


@pytest.fixture(scope="session")
def fuzzy_checker() -> FuzzyChecker:
    checker = FuzzyChecker()

    yield checker

    checker.write_output()
