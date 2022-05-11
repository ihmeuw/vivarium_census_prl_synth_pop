import numpy as np
import pandas as pd
from vivarium import Artifact
from vivarium.framework.engine import Builder
from vivarium.framework.population import PopulationView, SimulantData
from vivarium_public_health.utilities import to_years

from vivarium_census_prl_synth_pop import utilities
from vivarium_census_prl_synth_pop.constants import data_keys, metadata


class Population:

    @property
    def name(self):
        return "population"

    def setup(self, builder: Builder):
        self.config = builder.configuration.population
        self.randomness = builder.randomness.get_stream("household_sampling", for_initialization=True)

        self.columns_created = [
            'household_id',
            'address',  # TODO: ask rajan / james about adding a zipcode
            'relation_to_household_head',
            'sex', 
            'age',
            'race_ethnicity',
            'alive',
            'entrance_time',
            'exit_time'
        ]
        self.register_simulants = builder.randomness.register_simulants
        self.population_view = self._get_population_view(builder)
        self.population_data = self._load_population_data(builder)
        self._register_simulant_initializer(builder)

        builder.event.register_listener("time_step", self.on_time_step)  # TODO: is this the correct priority?

    def _register_simulant_initializer(self, builder: Builder) -> None:
        builder.population.initializes_simulants(
            self.generate_base_population,
            creates_columns=self.columns_created,
            requires_columns=['tracked'],
            )

    def _get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view(self.columns_created + ['tracked'])

    def _load_population_data(self, builder: Builder):
        households = builder.data.load(data_keys.POPULATION.HOUSEHOLDS)
        persons = builder.data.load(data_keys.POPULATION.PERSONS)
        return {'households': households, 'persons': persons}

    def generate_base_population(self, pop_data: SimulantData) -> None:
        # oversample households
        overshoot_idx = pd.Index(range(self.config.population_size))
        chosen_households = self.randomness.choice(
            index=overshoot_idx,
            choices=self.population_data['households']['census_household_id'],
            p=self.population_data['households']['household_weight']
        )
        # create unique id for resampled households
        chosen_households = pd.DataFrame({
            'census_household_id': chosen_households,
            'household_id': [
                idn + str(num) for (idn, num) in zip(chosen_households, range(len(chosen_households)))
            ]
        })
        # get all simulants per household
        chosen_persons = pd.merge(
            chosen_households,
            self.population_data['persons'],
            on='census_household_id',
            how='left'
        )

        # get rid simulants in excess of desired pop size
        households_to_discard = chosen_persons.loc[self.config.population_size:, 'household_id'].unique()
        chosen_persons = chosen_persons.query(f"household_id not in {list(households_to_discard)}")

        # drop non-unique household_id
        chosen_persons = chosen_persons.drop(columns='census_household_id')

        # format
        n_chosen = chosen_persons.shape[0]
        chosen_persons['address'] = 'NA'
        chosen_persons['entrance_time'] = pop_data.creation_time
        chosen_persons['exit_time'] = pd.NaT
        chosen_persons['alive'] = 'alive'
        chosen_persons['tracked'] = True

        # add back in extra simulants to reach desired pop size
        remainder = self.config.population_size - n_chosen
        if remainder > 0:
            extras = pd.DataFrame(
                data={
                    'household_id': ['NA'],
                    'address': ['NA'],
                    'age': [np.NaN],
                    'relation_to_household_head': ['NA'],
                    'sex': ['NA'],
                    'race_ethnicity': ['NA'],
                    'entrance_time': [pd.NaT],
                    'exit_time': [pd.NaT],
                    'alive': ['alive'],
                    'tracked': [False],
                },
                index=range(remainder)
            )
            chosen_persons = pd.concat([chosen_persons, extras])

        chosen_persons['age'] = chosen_persons['age'].astype('float64')
        chosen_persons = chosen_persons.set_index(pop_data.index)
        self.population_view.update(
            chosen_persons
        )

    def on_time_step(self, event):
        """Ages simulants each time step.

        Parameters
        ----------
        event : vivarium.framework.event.Event

        """
        population = self.population_view.get(event.index, query="alive == 'alive'")
        population["age"] += to_years(event.step_size)
        self.population_view.update(population)
