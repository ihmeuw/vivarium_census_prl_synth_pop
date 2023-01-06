import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event

from vivarium_census_prl_synth_pop.constants import metadata, paths


class PersonEmigration:
    """
    Handles migration of individuals (not in household groups) from within the US to outside of it.

    There are two types of individual moves in international emigration:
    - GQ person moves, where someone living in GQ emigrates.
    - Non-reference-person moves, where a non-reference-person living in a non-GQ household emigrates.

    Note that the names of these move types refer to the living situation *before* the move
    is complete.
    There is no overlap in the population at risk between these move types.
    """

    def __repr__(self) -> str:
        return "PersonEmigration()"

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return "person_emigration"

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self.name)
        self.columns_needed = [
            "relation_to_household_head",
            "housing_type",
            "exit_time",
            "tracked",
        ]
        self.population_view = builder.population.get_view(self.columns_needed)

        non_reference_person_move_rates_data = pd.read_csv(
            paths.NON_REFERENCE_PERSON_EMIGRATION_RATES_PATH,
        )

        non_reference_person_move_rates_lookup_table = builder.lookup.build_table(
            data=non_reference_person_move_rates_data,
            key_columns=["sex", "race_ethnicity", "state", "born_in_us"],
            parameter_columns=["age"],
            value_columns=["non_reference_person_emigration_rate"],
        )
        self.non_reference_person_move_rates = builder.value.register_rate_producer(
            f"{self.name}.move_rates",
            source=non_reference_person_move_rates_lookup_table,
        )

        builder.event.register_listener(
            "time_step",
            self.on_time_step,
            priority=metadata.PRIORITY_MAP["person_emigration.on_time_step"],
        )

    ########################
    # Event-driven methods #
    ########################

    def on_time_step(self, event: Event) -> None:
        """
        Determines which simulants will emigrate with each move type
        and removes them from the tracked population.
        """
        pop = self.population_view.get(
            event.index,
            query="alive == 'alive' and tracked == True",
        )

        non_reference_people_idx = pop.index[
            (pop["housing_type"] == "Standard")
            & (pop["relation_to_household_head"] != "Reference person")
        ]
        non_reference_person_movers_idx = self.randomness.filter_for_rate(
            non_reference_people_idx,
            self.non_reference_person_move_rates(non_reference_people_idx),
        )

        # TODO: GQ person moves

        # Leaving the US is equivalent to leaving the simulation
        pop.loc[non_reference_person_movers_idx, "exit_time"] = event.time
        pop.loc[non_reference_person_movers_idx, "tracked"] = False

        self.population_view.update(pop)
