import pandas as pd
import pytest

from vivarium_census_prl_synth_pop.components.population import Population

DOB_BEFORE_SIM_START = pd.Timestamp("1985-01-01 00:00:00")
DOB_AFTER_SIM_START = pd.Timestamp("2025-12-31 00:00:00")


fake_household_1 = pd.DataFrame(
    {
        "household_id": [1] * 5,
        "date_of_birth": [
            DOB_BEFORE_SIM_START,
            DOB_AFTER_SIM_START,
            DOB_AFTER_SIM_START,
            DOB_AFTER_SIM_START,
            DOB_BEFORE_SIM_START,
        ],
        "age": [39, 8, 5, 7, 35],
        "guardian_1": [-1, 0, 300, -1, -1],
        "relation_to_household_head": [
            "Opp-sex spouse",
            "Stepchild",
            "Adopted child",
            "Adopted child",
            "Other relative",
        ],
        "housing_type": ["Standard"] * 5,
    },
    index=[0, 1, 2, 3, 4],
)

fake_household_2 = pd.DataFrame(
    {
        "household_id": [2] * 5,
        "date_of_birth": [
            DOB_BEFORE_SIM_START,
            DOB_BEFORE_SIM_START,
            DOB_BEFORE_SIM_START,
            DOB_AFTER_SIM_START,
            DOB_BEFORE_SIM_START,
        ],
        "age": [68, 44, 7, 9, 38],
        "guardian_1": [-1, -1, 100, 100, -1],
        "relation_to_household_head": [
            "Parent-in-law",
            "Same-sex spouse",
            "Biological child",
            "Biological child",
            "Sibling",
        ],
        "housing_type": ["Standard"] * 5,
    },
    index=[5, 6, 7, 8, 9],
)

fake_household_3 = pd.DataFrame(
    {
        "household_id": [3] * 5,
        "date_of_birth": [
            DOB_BEFORE_SIM_START,
            DOB_BEFORE_SIM_START,
            DOB_BEFORE_SIM_START,
            DOB_BEFORE_SIM_START,
            DOB_AFTER_SIM_START,
        ],
        "age": [63, 27, 60, 23, 2],
        "guardian_1": [-1, -1, -1, -1, 35],
        "relation_to_household_head": [
            "Parent",
            "Sibling",
            "Parent",
            "Opp-sex spouse",
            "Biological child",
        ],
        "housing_type": ["Standard"] * 5,
    },
    index=[10, 11, 12, 13, 14],
)

fake_household_4 = pd.DataFrame(
    {
        "household_id": [4] * 5,
        "date_of_birth": [
            DOB_BEFORE_SIM_START,
            DOB_BEFORE_SIM_START,
            DOB_BEFORE_SIM_START,
            DOB_BEFORE_SIM_START,
            DOB_BEFORE_SIM_START,
        ],
        "age": [23, 22, 22, 21, 20],
        "guardian_1": [-1, -1, -1, -1, -1],
        "relation_to_household_head": [
            "Roommate",
            "Roommate",
            "Roommate",
            "Roommate",
            "Sibling",
        ],
        "housing_type": ["Standard"] * 5,
    },
    index=[15, 16, 17, 18, 19],
)

fake_household_5 = pd.DataFrame(
    {
        "household_id": [5] * 5,
        "date_of_birth": [
            DOB_BEFORE_SIM_START,
            DOB_AFTER_SIM_START,
            DOB_BEFORE_SIM_START,
            DOB_BEFORE_SIM_START,
            DOB_BEFORE_SIM_START,
        ],
        "age": [38, 5, 36, 34, 36],
        "guardian_1": [-1, 20, -1, 20, -1],
        "relation_to_household_head": [
            "Opp-sex spouse",
            "Other relative",
            "Roommate",
            "Other relative",
            "Other nonrelative",
        ],
        "housing_type": ["Standard"] * 5,
    },
    index=[20, 21, 22, 23, 24],
)


@pytest.fixture(scope="session")
def fake_population() -> pd.DataFrame:
    return pd.concat(
        [
            fake_household_1,
            fake_household_2,
            fake_household_3,
            fake_household_4,
            fake_household_5,
        ]
    )


def test_update_to_reference_person_and_relationships(mocker, fake_population):

    expected_relationships_1 = pd.Series(
        data=[
            "Reference person",
            "Biological child",
            "Adopted child",
            "Adopted child",
            "Other relative",
        ],
        index=[0, 1, 2, 3, 4],
    )
    expected_relationships_2 = pd.Series(
        data=[
            "Reference person",
            "Biological child",
            "Grandchild",
            "Grandchild",
            "Other relative",
        ],
        index=[5, 6, 7, 8, 9],
    )
    expected_relationships_3 = pd.Series(
        data=[
            "Reference person",
            "Biological child",
            "Other relative",
            "Child-in-law",
            "Grandchild",
        ],
        index=[10, 11, 12, 13, 14],
    )
    expected_relationships_4 = pd.Series(
        data=[
            "Reference person",
            "Roommate",
            "Roommate",
            "Roommate",
            "Roommate",
        ],
        index=[15, 16, 17, 18, 19],
    )
    expected_relationships_5 = pd.Series(
        data=[
            "Reference person",
            "Biological child",
            "Roommate",
            "Other relative",
            "Other nonrelative",
        ],
        index=[20, 21, 22, 23, 24],
    )
    expected_relationships = pd.concat(
        [
            expected_relationships_1,
            expected_relationships_2,
            expected_relationships_3,
            expected_relationships_4,
            expected_relationships_5,
        ]
    )

    pop_component = Population()

    # Setup class methods we need to update fake state table
    pop_component.start_time = pd.Timestamp("2020-04-01 00:00:00")
    pop_component.population_view = mocker.MagicMock()
    pop_component.population_view.get = lambda idx, query: fake_population.loc[idx]

    # This is a series with household_id as the index and the new reference person as the value
    expected_reference_person = (
        fake_population.loc[fake_population.index].groupby(["household_id"])["age"].idxmax()
    )
    updated_relationships = pop_component.get_updated_relation_to_reference_person(
        fake_population.index
    )

    assert not updated_relationships.isnull().any()
    assert (updated_relationships == expected_relationships).all()

    new_reference_person_idx = updated_relationships.index[
        updated_relationships == "Reference person"
    ]

    assert (expected_reference_person == new_reference_person_idx).all()
