import pandas as pd
import pytest

from vivarium_census_prl_synth_pop.components.population import Population
from vivarium_census_prl_synth_pop.constants import paths

fake_household_1 = pd.DataFrame(
    {
        "household_id": [1] * 5,
        "date_of_birth": [
            pd.Timestamp("1987-03-01 00:00:00"),
            pd.Timestamp("2023-05-11 00:00:00"),
            pd.Timestamp("2026-04-01 00:00:00"),
            pd.Timestamp("2025-11-21 00:00:00"),
            pd.Timestamp("1992-07-25 00:00:00"),
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
    }
)

fake_household_2 = pd.DataFrame(
    {
        "household_id": [2] * 5,
        "date_of_birth": [
            pd.Timestamp("1959-12-03 00:00:00"),
            pd.Timestamp("1983-01-31 00:00:00"),
            pd.Timestamp("1983-01-31 00:00:00"),
            pd.Timestamp("2025-04-30 00:00:00"),
            pd.Timestamp("1990-01-11 00:00:00"),
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
    }
)

fake_household_3 = pd.DataFrame(
    {
        "household_id": [3] * 5,
        "date_of_birth": [
            pd.Timestamp("1965-11-08 00:00:00"),
            pd.Timestamp("1992-05-18 00:00:00"),
            pd.Timestamp("1967-02-05 00:00:00"),
            pd.Timestamp("1995-04-28 00:00:00"),
            pd.Timestamp("2028-08-13 00:00:00"),
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
    }
)

fake_household_4 = pd.DataFrame(
    {
        "household_id": [4] * 5,
        "date_of_birth": [
            pd.Timestamp("1995-04-22 00:00:00"),
            pd.Timestamp("1996-07-02 00:00:00"),
            pd.Timestamp("1996-04-22 00:00:00"),
            pd.Timestamp("1996-09-16 00:00:00"),
            pd.Timestamp("1997-10-30 00:00:00"),
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
    }
)

fake_household_5 = pd.DataFrame(
    {
        "household_id": [5] * 5,
        "date_of_birth": [
            pd.Timestamp("1988-12-22 00:00:00"),
            pd.Timestamp("2025-04-11 00:00:00"),
            pd.Timestamp("1990-02-14 00:00:00"),
            pd.Timestamp("1992-08-15 00:00:00"),
            pd.Timestamp("1990-04-01 00:00:00"),
        ],
        "age": [38, 5, 36, 34, 36],
        "guardian_1": [-1, 20, -1, -1, -1],
        "relation_to_household_head": [
            "Opp-sex spouse",
            "Biological child",
            "Roommate",
            "Other relative",
            "Other nonrelative",
        ],
        "housing_type": ["Standard"] * 5,
    }
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
    ).reset_index()


def test_update_to_reference_person_and_relationships(fake_population):

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

    pop = Population()

    # Setup class methods we need to update fake state table
    pop.start_time = pd.Timestamp("2020-04-01 00:00:00")
    pop.reference_person_update_relationships_map = pd.read_csv(
        paths.REFERENCE_PERSON_UPDATE_RELATIONSHIP_DATA_PATH,
    )
    # This is a series with household_id as the index and the new reference person as the value
    expected_reference_person = (
        fake_population.loc[fake_population.index].groupby(["household_id"])["age"].idxmax()
    )
    updated_population = pop.update_reference_person_and_relationships(fake_population)

    assert not updated_population["relation_to_household_head"].isnull().any()
    assert (updated_population["relation_to_household_head"] == expected_relationships).all()

    new_reference_person_idx = updated_population.index[
        updated_population["relation_to_household_head"] == "Reference person"
    ]

    assert (expected_reference_person == new_reference_person_idx).all()
