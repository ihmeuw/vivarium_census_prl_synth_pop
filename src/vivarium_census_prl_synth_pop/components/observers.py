from pathlib import Path

from vivarium.framework.engine import Builder
from vivarium.framework.event import Event

from vivarium_cluster_tools import mkdir

from vivarium_census_prl_synth_pop.constants import metadata


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
        self.current_year = self.start_date.year
        self.clock = builder.time.clock()
        self.counter = 0
        self.output_path = self._build_output_root(builder) / "state_table.hdf"
        self.decennial_path = self._build_output_root(builder) / "decennial_census.hdf"
        self.population_view = builder.population.get_view(columns=[])

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
            if self.clock().day < 29: #because we only want one observation in April
                self.decennial_census(event, hdf_key=f"year_{self.clock().year}")

    def on_simulation_end(self, event: Event) -> None:
        end_date_str = f"ymd_{self.end_date.year}_{self.end_date.month}_{self.end_date.day}"
        state_table = self.population_view.get(event.index)
        state_table.to_hdf(self.output_path, end_date_str)

    def decennial_census(self, event: Event, hdf_key) -> None:
        pop = self.population_view.subview(metadata.DECENNIAL_CENSUS_COLUMNS_USED).get(event.index)
        pop["middle_initial"] = pop["middle_name"].astype(str).str[0]
        pop = pop.drop(columns="middle_name")
        pop.to_hdf(self.decennial_path, hdf_key)


    ###########
    # Helpers #
    ###########

    @staticmethod
    def _build_output_root(builder: Builder) -> Path:
        results_root = builder.configuration.output_data.results_directory
        output_root = Path(results_root) / "population_table"
        mkdir(output_root, exists_ok=True)
        return output_root
