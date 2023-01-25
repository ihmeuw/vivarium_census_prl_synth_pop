from vivarium.framework.engine import Builder

from vivarium_census_prl_synth_pop.constants import data_keys


class Immigration:
    """
    Handles migration of individuals *into* the US.
    """

    def __repr__(self) -> str:
        return "Immigration()"

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return "immigration"

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder):
        persons_data = builder.data.load(data_keys.POPULATION.PERSONS)
        households_data = builder.data.load(data_keys.POPULATION.HOUSEHOLDS)

        immigrants = persons_data[persons_data["immigrated_in_last_year"]]

        gq_households = households_data[households_data["household_type"] != "Housing unit"]
        is_gq = immigrants["census_household_id"].isin(gq_households["census_household_id"])
        self.gq_immigrants = immigrants[is_gq]

        non_gq_immigrants = immigrants[~is_gq]
        immigrant_reference_people = non_gq_immigrants[
            non_gq_immigrants["relation_to_household_head"] == "Reference person"
        ]

        is_household_immigrant = non_gq_immigrants["census_household_id"].isin(
            immigrant_reference_people["census_household_id"]
        )

        self.household_immigrants = non_gq_immigrants[is_household_immigrant]
        self.non_reference_person_immigrants = non_gq_immigrants[~is_household_immigrant]

        # Get the *household* (not person) weights for each household that can immigrate
        # in a household move, for use in sampling.
        self.immigrant_household_weights = households_data.set_index(
            "census_household_id"
        ).loc[
            immigrant_reference_people["census_household_id"],
            "household_weight",
        ]
