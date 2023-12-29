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

    This is a class so that diagnostics for an entire test run can be tracked,
    and output to a file at the end of the run.
    """

    def __init__(self) -> None:
        self.proportion_test_diagnostics = []

    def fuzzy_assert_proportion(
        self,
        name: str,
        observed_numerator: int,
        observed_denominator: int,
        target_proportion: Union[Tuple[float, float], float],
        fail_bayes_factor_cutoff: float = 100.0,
        inconclusive_bayes_factor_cutoff: float = 0.1,
        bug_issue_beta_distribution_parameters: Tuple[float, float] = (0.5, 0.5),
        name_additional: str = "",
    ) -> None:
        """
        Assert that an observed proportion of events came from a target distribution
        of proportions.
        This method performs a Bayesian hypothesis test between beta-binomial
        distributions based on the target (no bug/issue) and a "bug/issue" distribution
        and raises an AssertionError if the test decisively favors the "bug/issue" distribution.
        It warns, but does not fail, if the test is not conclusive (which usually
        means a larger population size is needed for a conclusive result),
        and gives an additional warning if the test could *never* be conclusive at this sample size.

        See more detail about the statistics used here:
        https://vivarium-research.readthedocs.io/en/latest/model_design/vivarium_features/automated_v_and_v/index.html#proportions-and-rates

        :param name:
            The name of the assertion, for use in messages and diagnostics.
            All assertions with the same name will output identical warning messages,
            which means pytest will aggregate those warnings.
        :param observed_numerator:
            The observed number of events.
        :param observed_denominator:
            The number of opportunities there were for an event to be observed.
        :param target_proportion:
            What the proportion of events / opportunities *should* be if there is no bug/issue
            in the simulation, as the number of opportunities goes to infinity.
            If this parameter is a tuple of two floats, they are interpreted as the 2.5th percentile
            and the 97.5th percentile of the uncertainty interval about this value.
            If this parameter is a single float, it is interpreted as an exact value (no uncertainty).
            Setting this target distribution is a research task; there is much more guidance on
            doing so at https://vivarium-research.readthedocs.io/en/latest/model_design/vivarium_features/automated_v_and_v/index.html#interpreting-the-hypotheses
        :param fail_bayes_factor_cutoff:
            The Bayes factor above which a hypothesis test is considered to favor a bug/issue so strongly
            that the assertion should fail.
            This cutoff trades off sensitivity with specificity and should be set in consultation with research;
            this is described in detail at https://vivarium-research.readthedocs.io/en/latest/model_design/vivarium_features/automated_v_and_v/index.html#sensitivity-and-specificity
            The default of 100 is conventionally called a "decisive" result in Bayesian hypothesis testing.
        :param inconclusive_bayes_factor_cutoff:
            The Bayes factor above which a hypothesis test is considered to be inconclusive, not
            ruling out a bug/issue.
            This will cause a warning.
            The default of 0.1 represents what is conventionally considered "substantial" evidence in
            favor of no bug/issue.
        :param bug_issue_beta_distribution_parameters:
            The parameters of the beta distribution characterizing our subjective belief about what
            proportion would occur if there was a bug/issue in the simulation, as the sample size goes
            to infinity.
            Defaults to a Jeffreys prior, which has a decent amount of mass on the entire interval (0, 1) but
            more mass around 0 and 1.
            Generally the default should be used in most circumstances; changing it is probably a
            research decision.
        :param name_additional:
            An optional additional name attribute that will be output in diagnostics but not in warnings.
            Useful for e.g. specifying the timestep when an assertion happened.
        """
        if isinstance(target_proportion, tuple):
            target_lower_bound, target_upper_bound = target_proportion
        else:
            target_lower_bound = target_upper_bound = target_proportion

        assert observed_numerator <= observed_denominator
        assert target_upper_bound >= target_lower_bound

        bug_issue_alpha, bug_issue_beta = bug_issue_beta_distribution_parameters
        bug_issue_distribution = scipy.stats.betabinom(
            a=bug_issue_alpha, b=bug_issue_beta, n=observed_denominator
        )

        if target_lower_bound == target_upper_bound:
            no_bug_issue_distribution = scipy.stats.binom(
                p=target_lower_bound, n=observed_denominator
            )
        else:
            a, b = self._fit_beta_distribution_to_ui(target_lower_bound, target_upper_bound)

            no_bug_issue_distribution = scipy.stats.betabinom(
                a=a, b=b, n=observed_denominator
            )

        bayes_factor = self._calculate_bayes_factor(
            observed_numerator, bug_issue_distribution, no_bug_issue_distribution
        )

        observed_proportion = observed_numerator / observed_denominator
        reject_null = bayes_factor > fail_bayes_factor_cutoff
        self.proportion_test_diagnostics.append(
            {
                "name": name,
                "name_addl": name_additional,
                "observed_proportion": observed_proportion,
                "observed_numerator": observed_numerator,
                "observed_denominator": observed_denominator,
                "target_lower_bound": target_lower_bound,
                "target_upper_bound": target_upper_bound,
                "bayes_factor": bayes_factor,
                "reject_null": reject_null,
            }
        )

        if reject_null:
            if observed_proportion < target_lower_bound:
                raise AssertionError(
                    f"{name} value {observed_proportion:g} is significantly less than expected, bayes factor = {bayes_factor:g}"
                )
            else:
                raise AssertionError(
                    f"{name} value {observed_proportion:g} is significantly greater than expected, bayes factor = {bayes_factor:g}"
                )

        if (
            target_lower_bound > 0
            and self._calculate_bayes_factor(
                0, bug_issue_distribution, no_bug_issue_distribution
            )
            < fail_bayes_factor_cutoff
        ):
            warnings.warn(
                f"Sample size too small to ever find that the simulation's '{name}' value is less than expected."
            )

        if target_upper_bound < 1 and (
            self._calculate_bayes_factor(
                observed_denominator, bug_issue_distribution, no_bug_issue_distribution
            )
            < fail_bayes_factor_cutoff
        ):
            warnings.warn(
                f"Sample size too small to ever find that the simulation's '{name}' value is greater than expected."
            )

        if fail_bayes_factor_cutoff > bayes_factor > inconclusive_bayes_factor_cutoff:
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
            return np.finfo(float).max

    @cache
    def _fit_beta_distribution_to_ui(
        self, lower_bound: float, upper_bound: float
    ) -> Tuple[float, float]:
        assert lower_bound > 0 and upper_bound < 1

        # Inspired by https://stats.stackexchange.com/a/112671/
        def objective(x):
            # np.exp ensures they are always positive
            a, b = np.exp(x)
            dist = scipy.stats.beta(a=a, b=b)

            squared_error_lower = self._quantile_squared_error(dist, lower_bound, 0.025)
            squared_error_upper = self._quantile_squared_error(dist, upper_bound, 0.975)

            try:
                return squared_error_lower + squared_error_upper
            except FloatingPointError:
                return np.finfo(float).max

        # It is quite important to start with a reasonable guess.
        ui_midpoint = (lower_bound + upper_bound) / 2
        # TODO: Further refine these concentration values. As long as we get convergence
        # with one of them (for all the assertions we do), we're good -- but this specific
        # list may not get us to convergence for future assertions we add.
        for first_guess_concentration in [1_000, 100, 10]:
            optimization_result = scipy.optimize.minimize(
                objective,
                x0=[
                    np.log(ui_midpoint * first_guess_concentration),
                    np.log((1 - ui_midpoint) * first_guess_concentration),
                ],
            )
            # Sometimes it warns that it *may* not have found a good solution,
            # but the solution it found is very accurate.
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
            return np.finfo(float).max

    def save_diagnostic_output(self) -> None:
        """
        Save diagnostics for optional human inspection.
        Can be useful to get more information about warnings, or to prioritize
        areas to be more thorough in manual V&V.
        """
        output = pd.DataFrame(self.proportion_test_diagnostics)
        output.to_csv(
            Path(os.path.dirname(__file__))
            / "v_and_v_output/proportion_test_diagnostics.csv",
            index=False,
        )


@pytest.fixture(scope="session")
def fuzzy_checker() -> FuzzyChecker:
    checker = FuzzyChecker()

    yield checker

    checker.save_diagnostic_output()


def multiplicative_drift_to_bound_at_timestep(value, drift_per_timestep, num_timesteps):
    return value * pow(drift_per_timestep, num_timesteps)


def multiplicative_drift_to_bounds_at_timestep(
    value, lower_bound_drift_per_timestep, upper_bound_drift_per_timestep, num_timesteps
):
    return (
        multiplicative_drift_to_bound_at_timestep(
            value, lower_bound_drift_per_timestep, num_timesteps
        ),
        multiplicative_drift_to_bound_at_timestep(
            value, upper_bound_drift_per_timestep, num_timesteps
        ),
    )


def multiplicative_drift_to_bound_through_timestep(value, drift_per_timestep, num_timesteps):
    # Assumption: timesteps have equal sample sizes. This is reasonably accurate.
    return np.mean(
        [
            multiplicative_drift_to_bound_at_timestep(value, drift_per_timestep, x)
            for x in range(num_timesteps)
        ]
    )


def multiplicative_drift_to_bounds_through_timestep(
    value, lower_bound_drift_per_timestep, upper_bound_drift_per_timestep, num_timesteps
):
    return (
        multiplicative_drift_to_bound_through_timestep(
            value, lower_bound_drift_per_timestep, num_timesteps
        ),
        multiplicative_drift_to_bound_through_timestep(
            value, upper_bound_drift_per_timestep, num_timesteps
        ),
    )
