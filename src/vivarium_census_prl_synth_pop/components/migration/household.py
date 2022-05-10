import pandas as pd
import faker
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import PopulationView, SimulantData
from vivarium.framework.values import Pipeline
from vivarium_public_health.utilities import DAYS_PER_YEAR


class HouseholdMigration:
    """
    - on simulant_initialization, adds address to population table per household_id
    - on time_step, updates some households to new addresses

    ASSUMPTION:
    - households will always move to brand-new addresses (as opposed to vacated addresses)
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
        self.fake = faker.Faker()
        faker.Faker.seed(self.config.randomness.faker_seed)
        self.provider = faker.providers.address.en_US.Provider(faker.Generator())

        self.columns_needed = ['household_id', 'address']
        self.population_view = self._get_population_view(builder)
        self.household_ids = None

        self._register_simulant_initializer(builder)
        self._register_on_time_step_listener(builder)

    def _get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view(self.columns_needed)

    def _register_simulant_initializer(self, builder: Builder) -> None:
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            requires_columns=self.columns_needed,
        )

    def _register_on_time_step_listener(self, builder: Builder) -> None:
        builder.event.register_listener("time_step", self.on_time_step)

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """
        add addresses to each household in the population table
        create a datastructure holding all household ids
        """
        households = self.population_view.subview(['household_id']).get(pop_data.index)
        unique_household_ids = households.squeeze().drop_duplicates()
        self.household_ids = unique_household_ids
        address_map = {
            household_id: self._generate_single_fake_address() for household_id in unique_household_ids
        }
        households['address'] = households['household_id'].map(address_map)
        self.population_view.update(
            households
        )

    def on_time_step(self, event: Event):
        """
        choose which households move
        move those households to a new address
        """
        households = self.population_view.subview(['household_id','address']).get(event.index)
        households_that_move = self._determine_if_moving()
        households = households.query(f'household_id in {households_that_move}')
        new_address_map = {
            old_address: self._generate_single_fake_address() for old_address in households['address'].unique()
        }
        households['address'] = households['address'].map(new_address_map)
        self.population_view.update(
            households
        )

    ##################
    # Helper methods #
    ##################

    def _generate_single_fake_address(self):
        orig_address = self.fake.unique.address()
        address = orig_address.split('\n')[0]
        address += ', ' + orig_address.split('\n')[1].split(',')[0] + ', FL ' + self.provider.postcode_in_state('FL')
        return address

    def _determine_if_moving(self) -> list:
        households = pd.Series(self.household_ids)
        probability_moving = 0.15 * (self.config.time.step_size / DAYS_PER_YEAR)  # TODO: wrap '15' in lookup table
        probability_moving_per_household = len(households)*[probability_moving]
        households_that_move = self.randomness.filter_for_probability(
            households,
            probability_moving_per_household,
        )
        return list(households_that_move)
