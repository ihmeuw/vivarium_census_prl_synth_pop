import os
import warnings
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

    This is implemented as a class, where a single instance of the class is used
    throughout an entire run of the integration tests.
    This is because we want to apply a Bonferroni correction for testing multiple
    hypotheses across the entire "family" of hypotheses tested.
    """

    def __init__(
        self, num_comparisons: int, overall_significance_level: float, power_level: float
    ) -> None:
        self.num_comparisons = num_comparisons
        self.overall_significance_level = overall_significance_level
        self.power_level = power_level
        self.comparisons_made = []

    def fuzzy_assert_proportion(
        self,
        name: str,
        boolean_values: pd.Series,
        target_value: Optional[float] = None,
        target_value_min: Optional[float] = None,
        target_value_max: Optional[float] = None,
        name_addl: Optional[str] = "",
    ) -> None:
        if target_value is not None:
            target_value_min = target_value
            target_value_max = target_value

        if target_value_min is None or target_value_max is None:
            raise ValueError(
                f"{name}: Not enough information about the target value supplied"
            )

        assert target_value_max >= target_value_min

        # Bonferroni correction
        test_significance_level = self.overall_significance_level / self.num_comparisons

        numerator = boolean_values.sum()
        denominator = len(boolean_values)
        proportion = boolean_values.mean()

        # We can be dealing with some _extremely_ unlikely events here, so we have to set numpy to not error
        # if we generate a probability too small to be stored in a floating point number(!), which is known
        # as "underflow"
        # Technically this leads to some inaccuracy but this should be so miniscule as to not matter in practice
        with np.errstate(under="ignore"):
            p_value = self._two_tailed_binomial_test(
                numerator, denominator, target_value_min, target_value_max
            )

            # The maximum integer that is in the lower rejection region
            rejection_area_low_max = self._binary_search_integers(
                lambda x: self._two_tailed_binomial_test(
                    x, denominator, true_value_min, true_value_max
                ),
                test_significance_level,
                0,
                np.floor(true_value_min * denominator),
            )
            # The minimum integer that is in the higher rejection region
            rejection_area_high_min = (
                self._binary_search_integers(
                    # Negative of function and target so the function is monotonically *increasing*
                    # instead of decreasing over the range specified.
                    lambda x: -self._two_tailed_binomial_test(
                        x, denominator, true_value_min, true_value_max
                    ),
                    -test_significance_level,
                    np.ceil(true_value_max * denominator),
                    denominator,
                )
                + 1
            )
            assert rejection_area_low_max <= rejection_area_high_min

            # Find the maximum underlying simulation value we would be powered to detect as *lower*
            # than the target.
            if rejection_area_low_max <= 0:
                if true_value_min != 0:
                    warnings.warn(
                        f"Not enough statistical power to ever find that the simulation's '{name}' value is lower than {true_value_min}."
                    )
                powered_value_lb = 0
            else:
                powered_value_lb = self._binary_search_floats(
                    # Negative of function and target so the function is monotonically *increasing*
                    # instead of decreasing over the range specified.
                    lambda p: -(
                        scipy.stats.binom.cdf(rejection_area_low_max + 1, denominator, p)
                        + scipy.stats.binom.sf(rejection_area_high_min - 1, denominator, p)
                    ),
                    -self.power_level,
                    0,
                    true_value_min,
                )

            # Find the minimum underlying simulation value we would be powered to detect as *higher*
            # than the target.
            if rejection_area_high_min >= denominator:
                if true_value_max != 1:
                    warnings.warn(
                        f"Not enough statistical power to ever find that the simulation's '{name}' value is higher than {true_value_max}."
                    )
                powered_value_ub = 1
            else:
                powered_value_ub = self._binary_search_floats(
                    lambda p: (
                        scipy.stats.binom.cdf(rejection_area_low_max + 1, denominator, p)
                        + scipy.stats.binom.sf(rejection_area_high_min - 1, denominator, p)
                    ),
                    self.power_level,
                    true_value_max,
                    1,
                )

        reject_null = p_value < test_significance_level
        self.comparisons_made.append(
            {
                "name": name,
                "name_addl": name_addl,
                "proportion": proportion,
                "numerator": numerator,
                "denominator": denominator,
                "target_value_min": target_value_min,
                "target_value_max": target_value_max,
                "p_value": p_value,
                "test_significance_level": test_significance_level,
                "reject_null": reject_null,
                "rejection_area_low_max": rejection_area_low_max,
                "rejection_area_high_min": rejection_area_high_min,
                "powered_value_lb": powered_value_lb,
                "powered_value_ub": powered_value_ub,
            }
        )

        if reject_null:
            if boolean_values.mean() < target_value_min:
                raise AssertionError(
                    f"{name} value {proportion:g} is significantly less than {target_value_min:g}, p = {p_value:g} <= {test_significance_level:g}"
                )
            else:
                raise AssertionError(
                    f"{name} value {proportion:g} is significantly greater than {target_value_max:g}, p = {p_value:g} <= {test_significance_level:g}"
                )

    def write_output(self) -> None:
        output = pd.DataFrame(self.comparisons_made)
        output.to_csv(
            Path(os.path.dirname(__file__)) / "v_and_v_output/proportion_tests.csv",
            index=False,
        )

    def _two_tailed_binomial_test(
        self, numerator, denominator, null_hypothesis_min, null_hypothesis_max
    ):
        # Zeb note: This method is adapted from scipy.stats.binomtest, which can be found at:
        # https://github.com/scipy/scipy/blob/c1ed5ece8ffbf05356a22a8106affcd11bd3aee0/scipy/stats/_binomtest.py#L202-L333
        # All comments in this method that are not prefixed with "Zeb note" are copied from the original.
        # The key change I've made is that we allow the null hypothesis to be a range.
        # For the purposes of p-values, we treat the probability of a given result as being
        # the *maximum* probability of that result for *any possible* proportion value in the range.
        # This seems to be sort of like a minimax hypothesis test, though I am not sure what it is called.
        # More about the reasoning behind this change can be found here:
        # https://vivarium-research.readthedocs.io/en/latest/model_design/vivarium_features/automated_v_and_v/index.html#proportions-and-rates
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
            ix = self._binary_search_integers(
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
            ix = self._binary_search_integers(
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

    def _binary_search_integers(self, a, d, lo, hi):
        # Zeb note: This method is copied exactly from scipy.stats._binomtest._binary_search_for_binom_tst, which can be found at:
        # https://github.com/scipy/scipy/blob/c1ed5ece8ffbf05356a22a8106affcd11bd3aee0/scipy/stats/_binomtest.py#L336-L375
        # All comments in this method that are not prefixed with "Zeb note" are copied from the original.
        """
        Conducts an implicit binary search on a function specified by `a`.

        Meant to be used on the binomial PMF for the case of two-sided tests
        to obtain the value on the other side of the mode where the tail
        probability should be computed. The values on either side of
        the mode are always in order, meaning binary search is applicable.

        Parameters
        ----------
        a : callable
        The function over which to perform binary search. Its values
        for inputs lo and hi should be in ascending order.
        d : float
        The value to search.
        lo : int
        The lower end of range to search.
        hi : int
        The higher end of the range to search.

        Returns
        -------
        int
        The index, i between lo and hi
        such that a(i)<=d<a(i+1)
        """
        while lo < hi:
            mid = lo + (hi - lo) // 2
            midval = a(mid)
            if midval < d:
                lo = mid + 1
            elif midval > d:
                hi = mid - 1
            else:
                return mid
        if a(lo) <= d:
            return lo
        else:
            return lo - 1

    def _binary_search_floats(self, a, d, lo: float, hi: float, tol: float = 0.00001):
        # Copied from the above, but modified to work with floats
        """
        Conducts an implicit binary search on a function specified by `a`.

        Parameters
        ----------
        a : callable
        The function over which to perform binary search. Its values
        for inputs lo and hi should be in ascending order.
        d : float
        The value to search.
        lo : float
        The lower end of range to search.
        hi : float
        The higher end of the range to search.

        Returns
        -------
        float
        A value, i between lo and hi
        such that a(i) is within tol of d
        """
        iterations = 0
        while lo < hi:
            if iterations > 1_000:
                raise ValueError("tol is too small!")
            mid = lo + (hi - lo) / 2
            midval = a(mid)
            if np.abs(midval - d) < tol:
                return mid
            if midval < d:
                lo = mid
            else:
                hi = mid
            iterations += 1


@pytest.fixture(scope="session")
def fuzzy_checker(request) -> FuzzyChecker:
    checker = FuzzyChecker(
        # We need to supply the total number of comparisons we intend to make (and therefore
        # the total number of hypotheses we intend to test) so the appropriate Bonferroni
        # correction can be made.
        # NOTE: This will need to be updated when any new tests with fuzzy checks in them
        # are added.
        # The logic below will print out the correct updated value if the tests are run
        # with an outdated value.
        num_comparisons=601,
        # Maximum probability of getting any failure by chance when all the values are truly in the
        # ranges asserted (false alarm).
        # The lower this number is, the less sensitive we will be to true issues.
        # A number of simplifying assumptions are made, which make this value an upper bound:
        # the true probability of a random failure will be lower, with corresponding loss of
        # sensitivity.
        overall_significance_level=0.05,
        # Power level for the power tests that are performed along with each hypothesis test.
        # Note that these power tests, unlike the significance tests themselves, have no effect
        # on whether the tests pass or fail.
        power_level=0.80,
    )

    yield checker

    checker.write_output()

    comparisons_intended = checker.num_comparisons
    comparisons_made = len(checker.comparisons_made)
    if request.node.testsfailed > 0:
        # If any tests failed, there was likely an early-exit and not all of the comparisons
        # intended were actually made.
        # Running the exact equality assert in this case would just add noise and make someone think num_comparisons
        # needed to be updated when it didn't.
        assert_message = f"num_comparisons should be at least {comparisons_made} in the fuzzy checker, resolve other failures and then update the value in conftest.py"
        assert comparisons_made <= comparisons_intended, assert_message
    else:
        assert_message = f"num_comparisons should be {comparisons_made} in the fuzzy checker, update the value in conftest.py"
        assert comparisons_intended == comparisons_made, assert_message
