from vivarium.framework.engine import Builder
from vivarium.framework.population import PopulationView
from vivarium_public_health.population import Mortality as _Mortality

class Mortality(_Mortality):
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