from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import PopulationView
from vivarium_public_health.population import Mortality as _Mortality


class Mortality(_Mortality):
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.updated_relationship_to_reference_person = builder.value.get_value(
            "updated_relationship_to_reference_person"
        )

    def _get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view(
            [
                self.cause_of_death_column_name,
                self.years_of_life_lost_column_name,
                "alive",
                "exit_time",
                "age",
                "sex",
                "relationship_to_reference_person",
            ]
        )

    def on_time_step(self, event: Event) -> None:
        super().on_time_step(event)

        new_relationship_to_reference_person = self.updated_relationship_to_reference_person(
            event.index
        )
        self.population_view.update(new_relationship_to_reference_person)
