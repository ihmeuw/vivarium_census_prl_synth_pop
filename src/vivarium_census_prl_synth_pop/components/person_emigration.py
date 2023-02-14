from functools import partial

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import SimulantData

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
        self.household_details = builder.value.get_value("household_details")
        self.columns_needed = [
            "relation_to_household_head",
            "in_united_states",
            "exit_time",
            "tracked",
            "state_id_for_lookup",
        ]
        self.population_view = builder.population.get_view(self.columns_needed)
        self.updated_relation_to_reference_person = builder.value.get_value(
            "updated_relation_to_reference_person"
        )

        non_reference_person_move_rates_lookup_table = builder.lookup.build_table(
            data=pd.read_csv(paths.NON_REFERENCE_PERSON_EMIGRATION_RATES_PATH).rename(
                columns={"state": "state_id_for_lookup"}
            ),
            key_columns=["sex", "race_ethnicity", "state_id_for_lookup", "born_in_us"],
            parameter_columns=["age"],
            value_columns=["non_reference_person_emigration_rate"],
        )
        self.non_reference_person_move_rates = builder.value.register_rate_producer(
            f"{self.name}.non_reference_person_move_rates",
            source=partial(self.get_move_rates, non_reference_person_move_rates_lookup_table),
        )

        gq_person_move_rates_lookup_table = builder.lookup.build_table(
            data=pd.read_csv(
                paths.GQ_PERSON_EMIGRATION_RATES_PATH,
            ).rename(columns={"state": "state_id_for_lookup"}),
            key_columns=["sex", "race_ethnicity", "state_id_for_lookup", "born_in_us"],
            parameter_columns=["age"],
            value_columns=["gq_person_emigration_rate"],
        )
        self.gq_person_move_rates = builder.value.register_rate_producer(
            f"{self.name}.gq_person_move_rates",
            source=partial(self.get_move_rates, gq_person_move_rates_lookup_table),
        )

        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=["in_united_states"],
        )
        builder.event.register_listener(
            "time_step",
            self.on_time_step,
            priority=metadata.PRIORITY_MAP["person_emigration.on_time_step"],
        )

    def get_move_rates(self, lookup_table: LookupTable, idx: pd.Index) -> LookupTable:
        """The emigration rates lookup tables require a state_id column in the
        state table. Since the state_id column exists as part of the
        household_details pipeline, this method updates the state_id_for_lookup
        column with the pipeline values and then returns the lookup table
        """
        pop = self.population_view.get(idx)
        pop["state_id_for_lookup"] = self.household_details(idx)["state_id"]
        self.population_view.update(pop)
        return lookup_table(idx)

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pop_update = pd.DataFrame(
            {
                "in_united_states": True,
            },
            index=pop_data.index,
        )
        self.population_view.update(pop_update)

    def on_time_step(self, event: Event) -> None:
        """
        Determines which simulants will emigrate with each move type
        and removes them from the tracked population.
        """
        pop = self.population_view.get(
            event.index,
            query="alive == 'alive' and in_united_states == True and tracked == True",
        )
        household_details = self.household_details(pop.index)
        non_reference_people_idx = pop.index[
            (household_details["housing_type"] == "Standard")
            & (pop["relation_to_household_head"] != "Reference person")
        ]
        non_reference_person_movers_idx = self.randomness.filter_for_rate(
            non_reference_people_idx,
            self.non_reference_person_move_rates(non_reference_people_idx),
        )
        gq_people_idx = household_details.index[
            household_details["housing_type"] != "Standard"
        ]
        gq_person_movers_idx = self.randomness.filter_for_rate(
            gq_people_idx,
            self.gq_person_move_rates(gq_people_idx),
        )

        emigrating_idx = non_reference_person_movers_idx.union(gq_person_movers_idx)
        pop.loc[emigrating_idx, "in_united_states"] = False
        # Leaving the US is equivalent to leaving the simulation
        pop.loc[emigrating_idx, "exit_time"] = event.time
        pop.loc[emigrating_idx, "tracked"] = False

        self.population_view.update(pop)

        new_relation_to_ref_person = self.updated_relation_to_reference_person(event.index)
        self.population_view.update(new_relation_to_ref_person)
