from pathlib import Path

from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from vivarium_census_prl_synth_pop.constants import paths
from vivarium_cluster_tools import mkdir


class Observers:
    """
    at the start of simulant initialization:
    save population table with / key = date

    at the end of simulation:
    save population table with / key = date
    """

    def __repr__(self) -> str:
        return 'Observers()'

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
        self.counter = 0
        self.output_path = self._build_output_path(builder)
        self.population_view = builder.population.get_view(columns=[])

        builder.event.register_listener("time_step__prepare", self.on_time_step__prepare)
        builder.event.register_listener("simulation_end", self.on_simulation_end)

    def on_time_step__prepare(self, event: Event) -> None:
        if self.counter == 0:
            start_date_str = f"ymd_{self.start_date.year}_{self.start_date.month}_{self.start_date.day}"
            state_table = self.population_view.get(event.index)
            no_categories_state_table = self.type_category_to_object(state_table)
            no_categories_state_table.to_hdf(self.output_path, start_date_str)
        self.counter += 1

    def on_simulation_end(self, event: Event) -> None:
        end_date_str = f"ymd_{self.end_date.year}_{self.end_date.month}_{self.end_date.day}"
        state_table = self.population_view.get(event.index)
        no_categories_state_table = self.type_category_to_object(state_table)
        no_categories_state_table.to_hdf(self.output_path, end_date_str)

    ###########
    # Helpers #
    ###########

    def type_category_to_object(self, state_table):
        category_vars = state_table.columns[state_table.dtypes == 'category']
        for col in category_vars:
            state_table[col] = state_table[col].astype(object)
        return state_table

    @staticmethod
    def _build_output_path(builder: Builder) -> Path:
        results_root = builder.configuration.output_data.results_directory
        output_root = Path(results_root) / 'population_table'

        mkdir(output_root, exists_ok=True)

        input_draw = builder.configuration.input_data.input_draw_number
        seed = builder.configuration.randomness.random_seed
        # TODO: add back in 'scenario' if we add scenarios
        output_path = output_root / f'draw_{input_draw}_seed_{seed}.hdf'

        return output_path
