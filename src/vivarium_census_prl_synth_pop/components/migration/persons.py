# TODO: think through _when_ each todo needs to occur / create stubs accordingly
    # (could go thru each step in a cycle)
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.population import PopulationView


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

    _randomness_stream_name = 'person_migration'

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return "household_migration"

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self._randomness_stream_name)
        self.columns_needed = ['household_id','relationship_to_head_of_household']
        self.population_view = self._get_population_view(builder)

        self._register_on_time_step_listener(builder)


    def _get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view(self.columns_needed)

    def _register_on_time_step_listener(self, builder: Builder) -> None:
        builder.event.register_listener("time_step", self.on_time_step)

    def on_time_step(self) -> None:
        # TODO: pull all persons
        #  look up probability of moving
        #  determine which persons are moving
        #  choose a household at random
        #  move individual to new household
        #  update relationship to household head
        #  update population view
        persons = self.population_view['persons'].index
        probability_moving_per_person = self._probability_of_moving(persons)
        persons_who_move = self.randomness.filter_for_probability(
            persons,
            probability_moving_per_person,
        )
        households = self.population_view['households']['household_id']
        pass

    def _probability_of_moving(self, idx: pd.Index) -> list:
        # TODO: use lookup table instead of this
        return [0.15]*len(idx)

    def _choose_new_household(self):
        pass

