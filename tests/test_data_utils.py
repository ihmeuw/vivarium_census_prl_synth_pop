import pandas as pd
import pytest

from vivarium_census_prl_synth_pop.constants import metadata
from vivarium_census_prl_synth_pop.results_processing.formatter import (
    combine_joint_filers,
    flatten_data,
    format_1040_dataset,
)


@pytest.fixture(scope="module")
def dummy_1040():
    return pd.DataFrame(
        {
            "simulant_id": list(range(8)) * 2,
            "joint_filer": [
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                False,
            ]
            * 2,
            "relationship_to_reference_person": [
                "Reference person",
                "Opp-sex spouse",
                "Biological child",
                "Reference person",
                "Opp-sex spouse",
                "Reference person",
                "Roommate",
                "Opp-sex spouse",
            ]
            * 2,
            "household_id": [
                10,
                10,
                10,
                11,
                11,
                12,
                13,
                14,
            ]
            * 2,
            "tax_year": [2020] * 8 + [2021] * 8,
            "ssn_itin": ["123-45-6789"] * 16,
            "copy_ssn": ["000-00-0000"] * 16,
        }
    )


@pytest.fixture(scope="module")
def dummy_tax_dependents():
    return pd.DataFrame(
        {
            "simulant_id": [2, 103, 104, 105, 106, 107, 108, 109] * 2,
            "guardian_id": [0, 0, 0, 0, 0, 3, 3, 5] * 2,
            "favorite_food": [
                "Pizza",
                "Cookie",
                "Ice cream",
                "Cheeseburger",
                "Sandwich",
                "Salad",
                "Tacos",
                "Pasta",
                "Ramen",
                "Waffles",
                "Cookies",
                "Watermelon",
                "Nachos",
                "BBQ",
                "Bagel",
                "Grapes",
            ],
            "tax_year": [2020] * 8 + [2021] * 8,
            "first_name": ["Fake First name"] * 16,
            "last_name": ["Secret last name"] * 16,
            "ssn_itin": ["987-65-4321"] * 16,
            "copy_ssn": ["111-22-3333"] * 16,
        }
    )


def test_combine_joint_filers(dummy_1040):
    joint_1040 = combine_joint_filers(dummy_1040)

    assert set(joint_1040.columns) == set(
        [
            "simulant_id",
            "relationship_to_reference_person",
            "joint_filer",
            "household_id",
            "tax_year",
            "ssn_itin",
            "copy_ssn",
            "spouse_simulant_id",
            "spouse_relationship_to_reference_person",
            "spouse_joint_filer",
            "spouse_tax_year",
            "spouse_household_id",
            "spouse_ssn_itin",
            "spouse_copy_ssn",
        ]
    )
    joint_filer_ids = dummy_1040.loc[dummy_1040["joint_filer"] == True, "simulant_id"]
    # Joint filer ids should not be in simulant ids
    assert not bool(set(joint_1040["simulant_id"]) & set(joint_filer_ids))
    # Check we are returned correct number of rows... original data - joint filer rows
    assert len(dummy_1040) - len(joint_filer_ids) == len(joint_1040)


def test_flatten_data(dummy_tax_dependents):
    dependents_wide = flatten_data(
        data=dummy_tax_dependents,
        index_cols=["guardian_id", "tax_year"],
        rank_col="simulant_id",
        value_cols=["favorite_food"],
    )
    # Dependent and guardian ids should never overlap
    assert not bool(
        set(dependents_wide.reset_index()["guardian_id"])
        & set(dummy_tax_dependents["simulant_id"])
    )
    # The length of rows should be total guardian/tax year combinations which is 6
    assert len(dependents_wide) == 6
    # Guardian/simulant id 0 has 4 dependents which is the highest number of dependents
    # Make sure we do not have extra columns - more than 4 dependent. When only have one
    # "value" column from our pivot so we can assert there should be 4 columns for 4 dependents.
    assert len(dependents_wide.columns) == 4
    # Assert expected nans for depdents 2, 3, 4 columns - we have 3 guardians (0, 3, 5) with
    # 5, 2, and 1 dependents respectively. We expected dependent 2 column to have 2 nans, dependent
    # 3 and dependent 4 columns to have 4 nans.
    assert dependents_wide["2_favorite_food"].isna().sum() == 2
    for dependent in ["3", "4"]:
        assert dependents_wide[f"{dependent}_favorite_food"].isna().sum() == 4


def test_format_1040_dataset(dummy_1040, dummy_tax_dependents):

    obs_data = {
        metadata.DatasetNames.TAXES_1040: dummy_1040,
        metadata.DatasetNames.TAXES_DEPENDENTS: dummy_tax_dependents,
    }
    tax_1040 = format_1040_dataset(obs_data)

    # No joint filer should be in the formatted simulant_id column
    # We must check each year because of migration/joint filing
    for year in tax_1040["tax_year"].unique():
        year_df = tax_1040.loc[tax_1040["tax_year"] == year]
        assert not bool(set(year_df["simulant_id"]) & set(year_df["spouse_simulant_id"]))
    # Check formatted tax 1040 has necessary output columns
    # Note this is before we clense our data of extra columns
    for member_prefix in [
        "",
        "spouse_",
        "dependent_1_",
        "dependent_2_",
        "dependent_3_",
        "dependent_4_",
    ]:
        for metadata_col in ["ssn", "copy_ssn"]:
            member_col = member_prefix + metadata_col
            assert member_col in tax_1040.columns
