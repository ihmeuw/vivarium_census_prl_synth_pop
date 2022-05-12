import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData


class PersonMigration:
    """
    - needs to be able to update a household_id (to an existing id)
    - needs to be able to update a relationship to head of household (to other nonrelative)
    - on_time_step, needs to be able to look up probability of moving and determine if moving
    - needs to be able to choose a new household at random
    ASSUMPTIONS:
    - head of household never moves to new household_id
    """

    def __repr__(self) -> str:
        return 'PersonMigration()'

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return "person_migration"

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self.name)
        self.columns_needed = ['household_id', 'relation_to_household_head']
        self.population_view = builder.population.get_view(self.columns_needed)
        self.household_ids = None
        move_rate_data = builder.lookup.build_table(.15)
        self.person_move_rate = builder.value.register_rate_producer(f'{self.name}.move_rate', source=move_rate_data)

        builder.event.register_listener("time_step", self.on_time_step)
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            requires_columns=['household_id'],
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        households = self.population_view.subview(['household_id']).get(pop_data.index)
        unique_households = households.squeeze().drop_duplicates()
        self.household_ids = list(unique_households)

    def on_time_step(self, event: Event) -> None:
        """
        Determines which simulants will move to a new household
        Moves those simulants to a new household_id
        Assigns those simulants relationship to head of household 'Other nonrelative'
        """
        persons = self.population_view.get(event.index)
        non_household_heads = persons.query(f"relation_to_household_head != 'Reference person'")
        persons_who_move = self.randomness.filter_for_rate(
            non_household_heads,
            self.person_move_rate(non_household_heads.index)
        )
        persons_who_move['household_id'] = self._get_new_household_ids(persons_who_move)
        persons_who_move['relation_to_household_head'] = "Other nonrelative"
        self.population_view.update(
            persons_who_move
        )

    ##################
    # Helper methods #
    ##################

    def _get_new_household_ids(self, persons_who_move) -> pd.Series:
        new_household_ids = persons_who_move['household_id'].copy()
        for idx, person in persons_who_move.iterrows():
            extra_seed = 0
            while new_household_ids[idx] == person['household_id']:
                new_household_id = self.randomness.choice(
                    pd.Index([idx]),
                    self.household_ids,
                    additional_key=extra_seed
                )
                new_household_ids[idx] = new_household_id.iloc[0]
                extra_seed += 1
        return list(new_household_ids)
