import warnings
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


class FuzzyTest:
    def __init__(self, num_comparisons: int, overall_significance_level: int) -> None:
        self.num_comparisons = num_comparisons
        self.overall_significance_level = overall_significance_level
        self.comparisons_made = 0

    def fuzzy_assert_proportion(
        self,
        boolean_values: pd.Series,
        true_value: Optional[float] = None,
        true_value_min: Optional[float] = None,
        true_value_max: Optional[float] = None,
    ):
        if true_value is not None:
            true_value_min = true_value
            true_value_max = true_value

        if true_value_min is None or true_value_max is None:
            raise ValueError("Not enough information about the true value supplied")

        assert true_value_max >= true_value_min

        self.comparisons_made += 1

        sample_size = len(boolean_values)

        # Bonferroni correction
        test_significance_level = self.overall_significance_level / self.num_comparisons

        numerator = boolean_values.sum()
        proportion = boolean_values.mean()

        # We can be dealing with some _extremely_ unlikely events here, so we have to set numpy to not error
        # if we generate a probability too small to be stored in a floating point number(!), which is known
        # as "underflow"
        # Technically this leads to some inaccuracy but this should be so miniscule as to not matter in practice
        with np.errstate(under="ignore"):
            p_value = self._two_tailed_binomial_test(
                numerator, sample_size, true_value_min, true_value_max
            )

            if p_value < test_significance_level:
                if boolean_values.mean() < true_value_min:
                    raise AssertionError(
                        f"Value {proportion:g} is significantly less than {true_value_min:g}, p = {p_value:g} <= {test_significance_level:g}"
                    )
                else:
                    raise AssertionError(
                        f"Value {proportion:g} is significantly greater than {true_value_max:g}, p = {p_value:g} <= {test_significance_level:g}"
                    )

            # To see if we have enough power, we check whether the most extreme results possible
            # (none or all of the values being true) would be significant at this sample size/significance level.
            if true_value_min != 0:
                cannot_detect_too_low = (
                    self._two_tailed_binomial_test(0, sample_size, true_value_min, 0.999)
                    > test_significance_level
                )

                if cannot_detect_too_low:
                    warnings.warn(
                        f"Not enough statistical power to find that the simulation's value is lower than {true_value_min}."
                    )

            if true_value_max != 1:
                cannot_detect_too_high = (
                    self._two_tailed_binomial_test(
                        sample_size, sample_size, 0.001, true_value_max
                    )
                    > test_significance_level
                )

                if cannot_detect_too_high:
                    warnings.warn(
                        f"Not enough statistical power to find that the simulation's value is higher than {true_value_max}."
                    )

    def _two_tailed_binomial_test(
        self, numerator, denominator, null_hypothesis_min, null_hypothesis_max
    ):
        # Adapted from https://github.com/scipy/scipy/blob/c1ed5ece8ffbf05356a22a8106affcd11bd3aee0/scipy/stats/_binomtest.py#L202-L333
        # This is a totally home-grown way to make the null hypothesis a range.
        # Is this reasonable?
        binom = scipy.stats.binom

        # Zeb note: I don't understand rerr, but have left it from the original SciPy implementation
        rerr = 1 + 1e-7
        if (
            numerator >= null_hypothesis_min * denominator
            and numerator <= null_hypothesis_max * denominator
        ):
            # special case as shortcut, would also be handled by `else` below
            pval = 1.0
        elif numerator < null_hypothesis_min * denominator:
            d = binom.pmf(numerator, denominator, null_hypothesis_min)
            ix = scipy.stats._binomtest._binary_search_for_binom_tst(
                lambda x1: -binom.pmf(x1, denominator, null_hypothesis_max),
                -d * rerr,
                np.ceil(null_hypothesis_max * denominator),
                denominator,
            )
            # y is the number of terms between mode and n that are <= d*rerr.
            # ix gave us the first term where a(ix) <= d*rerr < a(ix-1)
            # if the first equality doesn't hold, y=n-ix. Otherwise, we
            # need to include ix as well as the equality holds. Note that
            # the equality will hold in very very rare situations due to rerr.
            y = (
                denominator
                - ix
                + int(d * rerr == binom.pmf(ix, denominator, null_hypothesis_max))
            )
            pval = binom.cdf(numerator, denominator, null_hypothesis_min) + binom.sf(
                denominator - y, denominator, null_hypothesis_max
            )
        else:
            d = binom.pmf(numerator, denominator, null_hypothesis_max)
            ix = scipy.stats._binomtest._binary_search_for_binom_tst(
                lambda x1: binom.pmf(x1, denominator, null_hypothesis_min),
                d * rerr,
                0,
                np.floor(null_hypothesis_min * denominator),
            )
            # y is the number of terms between 0 and mode that are <= d*rerr.
            # we need to add a 1 to account for the 0 index.
            # For comparing this with old behavior, see
            # tst_binary_srch_for_binom_tst method in test_morestats.
            y = ix + 1
            pval = binom.cdf(y - 1, denominator, null_hypothesis_min) + binom.sf(
                numerator - 1, denominator, null_hypothesis_max
            )

        pval = min(1.0, pval)

        return pval


@pytest.fixture(scope="session")
def fuzzy_tester() -> FuzzyTest:
    tester = FuzzyTest(
        # NOTE: This will need to be updated when any new tests with fuzzy asserts
        # are added.
        # Do not update this if it is not the only thing that is failing!
        # Early exits from failing tests can make the number of asserts actually performed
        # different than what it would be with passing tests.
        num_comparisons=708,
        # Probability of getting any failure by chance when all the values are truly in the
        # ranges asserted (false alarm).
        # The lower this number is, the less sensitive we will be to true issues.
        overall_significance_level=0.05,
    )

    yield tester

    assert tester.num_comparisons == tester.comparisons_made
