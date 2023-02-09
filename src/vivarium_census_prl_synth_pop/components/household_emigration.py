import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event

from vivarium_census_prl_synth_pop.constants import metadata, paths


class HouseholdEmigration:
    """
    Handles migration of households from within the US to outside of it.

    Rates depend on the demographics of the household's reference person as well as US state.
    They are in terms of household moves per household-year.
    All non-GQ households are at risk of an emigration event.
    """

    def __repr__(self) -> str:
        return "HouseholdEmigration()"

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return "household_emigration"

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self.name)

        emigration_rate_data = builder.lookup.build_table(
            data=pd.read_csv(
                paths.HOUSEHOLD_EMIGRATION_RATES_PATH,
            ),
            key_columns=["sex", "race_ethnicity", "state", "born_in_us"],
            parameter_columns=["age"],
            value_columns=["household_emigration_rate"],
        )
        self.household_move_rate = builder.value.register_rate_producer(
            f"{self.name}.move_rate", source=emigration_rate_data
        )

        self.population_view = builder.population.get_view(
            [
                "household_id",
                "relation_to_household_head",
                "in_united_states",
                "exit_time",
                "tracked",
            ]
        )

        builder.event.register_listener(
            "time_step",
            self.on_time_step,
            priority=metadata.PRIORITY_MAP["household_emigration.on_time_step"],
        )

    ########################
    # Event-driven methods #
    ########################

    def on_time_step(self, event: Event):
        """
        Choose which households emigrate, and make them untracked.
        """
        pop = self.population_view.get(
            event.index,
            query="alive == 'alive' and in_united_states == True and tracked == True",
        )
        reference_people = pop[pop["relation_to_household_head"] == "Reference person"]

        emigrating_reference_people = self.randomness.filter_for_rate(
            reference_people,
            self.household_move_rate(reference_people.index),
        )

        emigrating_idx = pop.index[
            pop["household_id"].isin(emigrating_reference_people["household_id"])
        ]
        pop.loc[emigrating_idx, "in_united_states"] = False
        # Leaving the US is equivalent to leaving the simulation
        pop.loc[emigrating_idx, "exit_time"] = event.time
        pop.loc[emigrating_idx, "tracked"] = False

        self.population_view.update(pop)
