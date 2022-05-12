import pandas as pd
import faker
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData


class HouseholdMigration:
    """
    - on simulant_initialization, adds address to population table per household_id
    - on time_step, updates some households to new addresses

    ASSUMPTION:
    - households will always move to brand-new addresses (as opposed to vacated addresses)
    """

    def __repr__(self) -> str:
        return 'HouseholdMigration()'

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
        self.randomness = builder.randomness.get_stream(self.name)
        self.fake = faker.Faker()
        faker.Faker.seed(self.config.randomness.random_seed)
        self.provider = faker.providers.address.en_US.Provider(faker.Generator())

        self.probability_household_moving_pipeline_name = "probability_of_household_moving"
        self.columns_needed = ['household_id', 'address']
        self.population_view = builder.population.get_view(self.columns_needed)
        move_rate_data = builder.lookup.build_table(.15)
        self.household_move_rate = builder.value.register_rate_producer(f'{self.name}.move_rate', source=move_rate_data)

        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            requires_columns=self.columns_needed,
        )
        builder.event.register_listener("time_step", self.on_time_step)

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """
        add addresses to each household in the population table
        """
        households = self.population_view.subview(['household_id']).get(pop_data.index)
        address_assignments = self._generate_addresses(list(households.drop_duplicates().squeeze()))
        households['address'] = households['household_id'].map(address_assignments)
        self.population_view.update(
            households
        )

    def on_time_step(self, event: Event):
        """
        choose which households move
        move those households to a new address
        """
        households = self.population_view.subview(['household_id', 'address']).get(event.index)
        households_that_move = self._determine_if_moving(households['household_id'])
        old_addresses = list(households.query(f'household_id in {households_that_move}')['address'].drop_duplicates())
        old_addresses_to_new = self._generate_addresses(old_addresses)
        households['address'] = households['address'].replace(old_addresses_to_new)
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

    def _generate_addresses(self, households: list):
        addresses = [self._generate_single_fake_address() for i in range(len(households))]
        return pd.Series(addresses, index=households)

    def _determine_if_moving(self, households: pd.Series) -> list:
        households = households.drop_duplicates()
        households_that_move = self.randomness.filter_for_rate(
            households,
            self.household_move_rate(households.index),
        )
        return list(households_that_move)
