import pandas as pd
import faker
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import PopulationView, SimulantData
from vivarium.framework.values import Pipeline
from vivarium_public_health.utilities import DAYS_PER_YEAR


# TODO: look at examples for how to organize these methods
class HouseholdMigration:
    """
    - on simulant_initialization, adds address to population table per household_id
    - on time_step, updates some households to new addresses

    ASSUMPTION:
    - households will always move to new addresses
    """
    """
    - creates a datastructure with household_id as index that holds address (maybe other things)
    - create pipeline that takes in a household_id and outputs a corresponding address <<<
    - on init_simulants, need to initialize addresses for each household in pop table
    - need a way to update new addresses on_time_step

    ASSUMPTIONS:
    - households will always move to a new address
    """

    def __repr__(self) -> str:
        return 'HouseholdMigration()'

    _randomness_stream_name = 'household_migration'

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
        self.config = builder.configuration
        self.randomness = builder.randomness.get_stream(self._randomness_stream_name)
        self.fake = faker.Faker()  # TODO: need to add seeds
        self.provider = faker.providers.address.en_US.Provider(faker.Generator())

        self.columns_needed = ['household_id']
        self.population_view = self._get_population_view(builder)
        self.households = None
        self.get_address_book_pipeline_name = 'get_address_book'
        self.address_book = self._get_address_book_pipeline(builder)

        self._register_simulant_initializer(builder)
        self._register_on_time_step_listener(builder)

    def _get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view(self.columns_needed)

    def _register_simulant_initializer(self, builder: Builder) -> None:
        builder.population.initializes_simulants(
            self.on_initialize_simulants,  # TODO: need to add randomness / give faker a seed
            requires_columns=self.columns_needed,
        )

    def _register_on_time_step_listener(self, builder: Builder) -> None:
        builder.event.register_listener("time_step", self.on_time_step)

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        # initialize datastructure holding addresses for each household in pop table
        household_ids = self.population_view.subview(['household_id']).get(pop_data.index).squeeze()
        households = pd.DataFrame({
            'address': [self._generate_single_fake_address() for _ in household_ids],
            'household_id': household_ids,
        }).set_index('household_id')
        self.households = households

    def on_time_step(self, event: Event):
        households_that_move = self._determine_if_moving()
        for household in households_that_move:
            self._modify_address(household, self._generate_single_fake_address())

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def _get_address_book_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.get_address_book_pipeline_name,
            source=lambda household_id: self.households.query(f'household_id in {list(household_id)}')['address'],
            requires_columns=self.columns_needed
        )

    ##################
    # Helper methods #
    ##################

    def _generate_single_fake_address(self):
        orig_address = self.fake.unique.address()
        address = orig_address.split('\n')[0]
        address += ', ' + orig_address.split('\n')[1].split(',')[0] + ', FL ' + self.provider.postcode_in_state('FL')
        return address

    def _modify_address(self, household_id, address) -> None:
        # TODO: check format of inputs is correct
        self.households.query(f'household_id in {list(household_id)}')['address'] = address

    def _determine_if_moving(self) -> pd.DataFrame:
        households = pd.Series(self.households.index)
        probability_moving = 0.15 * (self.config.time.step_size / DAYS_PER_YEAR)  # TODO: wrap in lookup table
        probability_moving_per_household = len(households)*[probability_moving]
        households_that_move = self.randomness.filter_for_probability(
            households,
            probability_moving_per_household,
        )
        return households_that_move

    def _get_probability_of_moving(self, builder: Builder) -> LookupTable:
        # TODO: figure out how to construct this
        prob = pd.DataFrame({
            'probability_per_year': [0.15]
        })
        return builder.build_table(prob, value_columns='probability_per_year')
