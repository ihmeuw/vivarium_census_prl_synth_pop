from typing import List

import pandas as pd
import faker
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.time import get_time_stamp

from vivarium_census_prl_synth_pop.constants import metadata, data_keys
from vivarium_census_prl_synth_pop.constants import data_values


class HouseholdMigration:
    """
    - on simulant_initialization, adds address to population table per household_id
    - on time_step, updates some households to new addresses

    ASSUMPTION:
    - households will always move to brand-new addresses (as opposed to vacated addresses)
    - puma will not change (pumas and zip codes currently unrelated)
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
        self.location = builder.data.load(data_keys.POPULATION.LOCATION)
        self.start_time = get_time_stamp(builder.configuration.time.start)
        self.randomness = builder.randomness.get_stream(self.name)
        self.fake = faker.Faker()
        faker.Faker.seed(self.config.randomness.random_seed)
        self.provider = faker.providers.address.en_US.Provider(faker.Generator())

        self.columns_created = ['address', 'zipcode']
        self.columns_needed = ['household_id', 'address', 'zipcode']
        self.population_view = builder.population.get_view(self.columns_needed)
        move_rate_data = builder.lookup.build_table(data_values.HOUSEHOLD_MOVE_RATE_YEARLY)
        self.household_move_rate = builder.value.register_rate_producer(f'{self.name}.move_rate', source=move_rate_data)

        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=self.columns_created,
            requires_columns=['household_id'],
        )
        builder.event.register_listener("time_step", self.on_time_step)

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """
        add addresses to each household in the population table
        """
        if pop_data.creation_time >= self.start_time:
            parent_ids = pop_data.user_data['parent_ids']
            mothers = self.population_view.subview(['household_id', 'address', 'zipcode']).get(parent_ids.unique())

            # assign the same address to the same household id
            new_births = pd.DataFrame(data={
                'parent_id': parent_ids
            }, index=pop_data.index)
            new_births = new_births.merge(mothers, left_on='parent_id', right_index=True)

            self.population_view.update(new_births[self.columns_created])

        else:
            households = self.population_view.subview(['household_id']).get(pop_data.index)
            address_assignments = self._generate_addresses(list(households.drop_duplicates().squeeze()))
            households['address'] = households['household_id'].map(address_assignments['address'])
            households['zipcode'] = households['household_id'].map(address_assignments['zipcode'])
            self.population_view.update(
                households
            )

    def on_time_step(self, event: Event):
        """
        choose which households move
        move those households to a new address
        """
        households = self.population_view.subview(['household_id', 'address', 'zipcode']).get(event.index)
        households_that_move = self._determine_if_moving(households['household_id'])
        new_addresses = self._generate_addresses(households_that_move)

        households.loc[households.household_id.isin(households_that_move), 'address'] = households.household_id.map(
            new_addresses['address']
        )
        households.loc[households.household_id.isin(households_that_move), 'zipcode'] = households.household_id.map(
            new_addresses['zipcode']
        )
        self.population_view.update(
            households
        )

    ##################
    # Helper methods #
    ##################

    def _generate_single_fake_address(self, state: str):
        orig_address = self.fake.unique.address()
        address = orig_address.split('\n')[0]
        address += ', ' + orig_address.split('\n')[1].split(',')[0] + ', ' + state
        return address

    def _generate_addresses(self, households: List[str]):
        state = metadata.US_STATE_ABBRV_MAP[self.location]
        addresses = [self._generate_single_fake_address(state) for i in range(len(households))]
        zipcodes = [self.provider.postcode_in_state(state) for i in range(len(households))]
        return pd.DataFrame({
            'address': addresses,
            'zipcode': zipcodes
        }, index=households)

    def _determine_if_moving(self, households: pd.Series) -> list:
        households = households.drop_duplicates()
        households_that_move = self.randomness.filter_for_rate(
            households,
            self.household_move_rate(households.index),
        )
        return list(households_that_move)
