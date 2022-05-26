import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import PopulationView
from vivarium_public_health.population import Mortality as _Mortality

class Mortality(_Mortality):

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder) -> None:
        self.sim_start = builder.configuration.time.start
        self.sim_end = builder.configuration.time.end
        self.random = self._get_randomness_stream(builder)
        self.clock = self._get_clock(builder)
        self.all_cause_mortality_rate = self._get_all_cause_mortality_rate(builder)
        self.cause_specific_mortality_rate = self._get_cause_specific_mortality_rate(builder)
        self.mortality_rate = self._get_mortality_rate(builder)
        self.life_expectancy = self._get_life_expectancy(builder)
        self.population_view = self._get_population_view(builder)

        self._register_simulant_initializer(builder)
        self._register_on_timestep_listener(builder)

    def _get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view(
            [
                self.cause_of_death_column_name,
                self.years_of_life_lost_column_name,
                "alive",
                "exit_time",
                "age",
                "sex",
            ]
        )

    def _get_all_cause_mortality_rate(self, builder: Builder) -> LookupTable:
        acmr_data = builder.data.load("cause.all_causes.cause_specific_mortality_rate")
        last_data_year = acmr_data.year_end.max()
        if self.sim_end.year + 1 > last_data_year:
            year_starts = np.arange(last_data_year, self.sim_end.year + 1)
            years_to_fill = pd.DataFrame({
                'year_start': year_starts,
                'year_end': year_starts+1,
            })
            year_to_broadcast = acmr_data.query(
                f"year_end == {last_data_year}"
            ).drop(columns=['year_start', 'year_end'])

            future_years = pd.merge(year_to_broadcast, years_to_fill, how='cross')
            acmr_data = pd.concat([acmr_data, future_years])

        return builder.lookup.build_table(
            acmr_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )
