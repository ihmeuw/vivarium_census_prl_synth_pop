from pathlib import Path

from vivarium.framework.engine import Builder
from vivarium.framework.event import Event

from vivarium_census_prl_synth_pop.constants import data_values, metadata


class Observers:
    """
    at the start of simulant initialization:
    save population table with / key = date

    at the end of simulation:
    save population table with / key = date
    """

    def __repr__(self) -> str:
        return "Observers()"

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return "observers"

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder):
        self.start_date = builder.configuration.time.start
        self.end_date = builder.configuration.time.end
        self.clock = builder.time.clock()
        self.counter = 0
        self.output_path = self._build_output_root(builder) / "state_table.hdf"
        self.decennial_path = self._build_output_root(builder) / "decennial_census.hdf"

        self.randomness = builder.randomness.get_stream(self.name)
        self.population_view = builder.population.get_view(columns=[])
        response_probability_decennial = builder.lookup.build_table(
            data=data_values.RESPONSE_PROBABILITY_DECENNIAL
        )
        self.response_probability_decennial = builder.value.register_value_producer(
            f"{self.name}.response_probability_decennial",
            source=response_probability_decennial,
        )

        builder.event.register_listener("time_step__prepare", self.on_time_step__prepare)
        builder.event.register_listener("simulation_end", self.on_simulation_end)

    def on_time_step__prepare(self, event: Event) -> None:
        if self.counter == 0:
            start_date_str = (
                f"ymd_{self.start_date.year}_{self.start_date.month}_{self.start_date.day}"
            )
            state_table = self.population_view.get(event.index)
            state_table.to_hdf(self.output_path, start_date_str)
        self.counter += 1
        if (self.clock().year % 10 == 0) & (self.clock().month == 4):
            if self.clock().day < 29:  # because we only want one observation in April
                self.decennial_census(event, hdf_key=f"year_{self.clock().year}")

    def on_simulation_end(self, event: Event) -> None:
        end_date_str = f"ymd_{self.end_date.year}_{self.end_date.month}_{self.end_date.day}"
        state_table = self.population_view.get(event.index)
        state_table.to_hdf(self.output_path, end_date_str)

    def decennial_census(self, event: Event, hdf_key) -> None:
        pop = self.population_view.subview(metadata.DECENNIAL_CENSUS_COLUMNS_USED).get(
            event.index
        )
        pop["middle_initial"] = pop["middle_name"].astype(str).str[0]
        pop = pop.drop(columns="middle_name")

        # we don't have a 100% census response rate:
        respondents = self.randomness.filter_for_probability(
            pop, self.response_probability_decennial(pop.index), "all_moving_households"
        )

        respondents.to_hdf(self.decennial_path, hdf_key)

    ###########
    # Helpers #
    ###########

    @staticmethod
    def _build_output_root(builder: Builder) -> Path:
        results_root = builder.configuration.output_data.results_directory
        output_root = Path(results_root) / "population_table"

        from vivarium_cluster_tools import mkdir

        mkdir(output_root, exists_ok=True)
        return output_root
