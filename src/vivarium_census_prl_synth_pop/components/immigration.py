from math import ceil, floor
from typing import Callable

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.utilities import from_yearly

from vivarium_census_prl_synth_pop.constants import data_keys, metadata
from vivarium_census_prl_synth_pop.utilities import (
    sample_acs_group_quarters,
    sample_acs_persons,
    sample_acs_standard_households,
)


class Immigration:
    """
    Handles migration of individuals *into* the US.
    """

    def __repr__(self) -> str:
        return "Immigration()"

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return "immigration"

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self.name)

        persons_data = builder.data.load(data_keys.POPULATION.PERSONS)
        households_data = builder.data.load(data_keys.POPULATION.HOUSEHOLDS)

        self.total_person_weight = persons_data["person_weight"].sum()

        immigrants = persons_data[persons_data["immigrated_in_last_year"]]

        gq_households = households_data[households_data["household_type"] != "Housing unit"]
        is_gq = immigrants["census_household_id"].isin(gq_households["census_household_id"])
        self.gq_immigrants = immigrants[is_gq]
        self.gq_immigrants_per_time_step = self._immigrants_per_time_step(
            self.gq_immigrants,
            builder.configuration,
        )

        non_gq_immigrants = immigrants[~is_gq]
        immigrant_reference_people = non_gq_immigrants[
            non_gq_immigrants["relation_to_household_head"] == "Reference person"
        ]

        is_household_immigrant = non_gq_immigrants["census_household_id"].isin(
            immigrant_reference_people["census_household_id"]
        )

        self.household_immigrants = non_gq_immigrants[is_household_immigrant]
        self.household_immigrants_per_time_step = self._immigrants_per_time_step(
            self.household_immigrants,
            builder.configuration,
        )
        self.non_reference_person_immigrants = non_gq_immigrants[~is_household_immigrant]
        self.non_reference_person_immigrants_per_time_step = self._immigrants_per_time_step(
            self.non_reference_person_immigrants,
            builder.configuration,
        )

        # Get the *household* (not person) weights for each household that can immigrate
        # in a household move, for use in sampling.
        self.household_immigrant_households = households_data[
            households_data["census_household_id"].isin(
                immigrant_reference_people["census_household_id"]
            )
        ]
        # FIXME -- this can go away once state and PUMA are in the households pipeline, because
        # we will no longer need household rows to join to individuals
        self.non_reference_person_immigrant_households = households_data[
            households_data["census_household_id"].isin(
                self.non_reference_person_immigrants["census_household_id"]
            )
        ]
        self.gq_immigrant_households = households_data[
            households_data["census_household_id"].isin(
                self.gq_immigrants["census_household_id"]
            )
        ]

        self.simulant_creator = builder.population.get_simulant_creator()

        builder.event.register_listener(
            "time_step",
            self.on_time_step,
            priority=metadata.PRIORITY_MAP["immigration.on_time_step"],
        )

    ########################
    # Event-driven methods #
    ########################

    def on_time_step(self, event: Event):
        """
        Creates new simulants from immigration.

        Parameters
        ----------
        event : vivarium.population.PopulationEvent
            The event that triggered the function call.
        """

        # Create GQ immigrants
        self._create_individual_immigrants(
            self.gq_immigrants_per_time_step,
            sample_acs_group_quarters,
            self.gq_immigrant_households,
            self.gq_immigrants,
            "gq_immigrants",
            event.index,
        )

        # Create non-reference-person immigrants
        self._create_individual_immigrants(
            self.non_reference_person_immigrants_per_time_step,
            sample_acs_persons,
            self.non_reference_person_immigrant_households,
            self.non_reference_person_immigrants,
            "non_reference_person_immigrants",
            event.index,
        )

        num_household_immigrants = self._round_stochastically(
            self.household_immigrants_per_time_step, "num_household_immigrants"
        )

        if num_household_immigrants > 0:
            chosen_households, chosen_persons = sample_acs_standard_households(
                num_household_immigrants,
                self.household_immigrant_households,
                self.household_immigrants,
                randomness=self.randomness,
            )
            self.simulant_creator(
                len(chosen_persons),
                population_configuration={
                    "sim_state": "time_step",
                    "creation_type": "household_immigrants",
                    "acs_households": chosen_households,
                    "acs_persons": chosen_persons,
                    "current_population_index": event.index,
                    # Fertility component in VPH depends on this being present: https://github.com/ihmeuw/vivarium_public_health/blob/58485f1206a7b85b6d2aac3185ce71600fef6e60/src/vivarium_public_health/population/add_new_birth_cohorts.py#L195-L198
                    "parent_ids": -1,
                },
            )

    ##################
    # Helper methods #
    ##################

    def _immigrants_per_time_step(self, immigrants, configuration):
        immigrants_per_year = (
            # We rescale the proportion between immigrant population and total population to the
            # simulation's initial population size.
            # This value will not change over time during the simulation.
            (immigrants["person_weight"].sum() / self.total_person_weight)
            * configuration.population.population_size
        )
        return from_yearly(
            immigrants_per_year, pd.Timedelta(days=configuration.time.step_size)
        )

    def _create_individual_immigrants(
        self,
        expected_num: float,
        sampling_function: Callable,
        acs_households: pd.DataFrame,
        acs_persons: pd.DataFrame,
        creation_type: str,
        current_population_index: pd.Index,
    ) -> None:
        num_immigrants = self._round_stochastically(expected_num, f"num_{creation_type}")

        if num_immigrants > 0:
            chosen_persons = sampling_function(
                num_immigrants,
                acs_households,
                acs_persons,
                randomness=self.randomness,
            )
            self.simulant_creator(
                len(chosen_persons),
                population_configuration={
                    "sim_state": "time_step",
                    "creation_type": creation_type,
                    "acs_persons": chosen_persons,
                    "current_population_index": current_population_index,
                    # Fertility component in VPH depends on this being present: https://github.com/ihmeuw/vivarium_public_health/blob/58485f1206a7b85b6d2aac3185ce71600fef6e60/src/vivarium_public_health/population/add_new_birth_cohorts.py#L195-L198
                    "parent_ids": -1,
                },
            )

    def _round_stochastically(self, num: float, additional_key: str) -> int:
        """
        Implements stochastic rounding.
        For example, a value of 2.23 will round up to 3 with 23% probability, and down to 2
        the rest of the time.
        If we used traditional rounding, a value of e.g. 0.4 would round down to 0 on every
        single timestep, and that immigration event would *never* happen.
        """
        # We have to make a dummy index to use get_draw for a single float
        random_draw = self.randomness.get_draw(
            pd.Index([0]), additional_key=additional_key
        ).loc[0]

        round_up = (num % 1) > random_draw
        if round_up:
            return ceil(num)
        else:
            return floor(num)
