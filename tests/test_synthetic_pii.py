import string
from typing import List

import numpy as np
import pandas as pd
import pytest
from vivarium.framework.randomness import RandomnessStream

from vivarium_census_prl_synth_pop.constants import data_keys, data_values
from vivarium_census_prl_synth_pop.results_processing.addresses import (
    get_city_map,
    get_mailing_address_map,
    get_street_details_map,
    get_zipcode_map,
)
from vivarium_census_prl_synth_pop.results_processing.names import (
    get_employer_name_map,
    get_given_name_map,
    get_last_name_map,
)
from vivarium_census_prl_synth_pop.results_processing.ssn_and_itin import (
    _load_ids,
    generate_ssns,
)

key = "test_synthetic_data_generation"
clock = lambda: pd.Timestamp("2020-09-01")
seed = 0
randomness = RandomnessStream(key=key, clock=clock, seed=seed)


def get_draw(self, index, additional_key=None) -> pd.Series:
    # Mock get draw function
    s = np.random.uniform(size=len(index))
    return pd.Series(data=s, index=index)


@pytest.fixture()
def given_names():
    names = pd.DataFrame(
        data={
            "sex": ["Female", "Female", "Male", "Male"] * 2,
            "yob": [1999, 1999, 1999, 1999, 2000, 2000, 2000, 2000],
            "name": ["Mary", "Annie", "Luis", "John", "Scarlett", "Anna", "Mark", "Brad"],
            "freq": [200, 100, 150, 50, 100, 25, 100, 150],
        }
    )
    names = names.set_index(["yob", "sex", "name", "freq"])
    return names


@pytest.fixture()
def last_names():
    last_names = pd.DataFrame(
        data={
            "name": ["Smith", "Johnson", "Jackson"],
            "rank": [1, 2, 3],
            "count": [2000, 1500, 1000],
            "prop100k": [20.01, 15.01, 10.01],
            "cum_prop100k": [20.01, 15.01, 10.01],
            "pctwhite": [70.01, 60.01, 50.01],
            "pctblack": [22.01, 21.01, 20.01],
            "pctapi": [0.55, 0.54, 0.53],
            "pctasian": [0.62, 0.61, 0.60],
            "pct2prace": [3.03, 3.02, 3.01],
            "pcthispanic": [2.02, 2.01, 2.00],
            "White": [0.34, 0.26, 0.40],
            "Latino": [0.10, 0.35, 0.55],
            "Black": [0.20, 0.40, 0.40],
            "Asian": [0.34, 0.33, 0.33],
            "Multiracial or Other": [0.11, 0.12, 0.13],
            "AIAN": [0.01, 0.02, 0.03],
            "NHOPI": [0.05, 0.04, 0.03],
        }
    )
    cols = list(last_names.columns)
    last_names = last_names.set_index(cols)

    return last_names


@pytest.fixture()
def fake_obs_data():
    size = 100_000
    return {
        "fake_observer": pd.DataFrame(
            {
                "first_name_id": list(range(size)),
                "middle_name_id": list(range(size)),
                "year_of_birth": [1999, 1999, 2000, 2000] * 25_000,
                "sex": ["Male", "Female"] * 50_000,
                "last_name_id": list(range(size)),
                "race_ethnicity": ["White", "Black", "Latino", "Asian"] * 25_000,
                "date_of_birth": [pd.Timestamp("1985-01-01 00:00:00")] * size,
            }
        )
    }


def get_name_frequency_proportions(
    names: pd.Series, population_demographics: pd.DataFrame, groupby_cols: List[str]
) -> pd.Series:
    name_totals = pd.concat([population_demographics, names], axis=1).rename(
        columns={0: "name"}
    )
    names_grouped = name_totals.groupby(groupby_cols)["name"].value_counts()
    grouped_totals = name_totals[groupby_cols].value_counts()
    name_proportions = names_grouped / grouped_totals

    return name_proportions


def test_first_and_middle_names(mocker, given_names, fake_obs_data):
    artifact = mocker.MagicMock()
    artifact.load.return_value = given_names

    # Get proportion of given names for tests
    totals = given_names.reset_index().groupby(["yob", "sex"])["freq"].sum()
    name_freq = given_names.reset_index("freq")
    proportions = name_freq["freq"] / totals
    proportions.index = proportions.index.rename("year_of_birth", "yob")

    # Get name frequencies for comparison
    first_names = get_given_name_map(
        "first_name_id",
        fake_obs_data,
        artifact,
        randomness,
        "1234",
    )
    first_name_proportions = get_name_frequency_proportions(
        first_names["first_name"],
        fake_obs_data["fake_observer"],
        ["year_of_birth", "sex"],
    )

    assert np.isclose(
        first_name_proportions.sort_index(), proportions.sort_index(), atol=1e-02
    ).all()

    other_seed_first_names = get_given_name_map(
        "first_name_id",
        fake_obs_data,
        artifact,
        randomness,
        "2345",
    )

    assert (first_names["first_name"] != other_seed_first_names["first_name"]).any()

    middle_names = get_given_name_map(
        "middle_name_id",
        fake_obs_data,
        artifact,
        randomness,
        "1234",
    )
    middle_name_proportions = get_name_frequency_proportions(
        middle_names["middle_name"], fake_obs_data["fake_observer"], ["year_of_birth", "sex"]
    )

    assert np.isclose(
        middle_name_proportions.sort_index(), proportions.sort_index(), atol=1e-02
    ).all()
    assert (first_names["first_name"] != middle_names["middle_name"]).any()


def test_last_names_proportions(mocker, last_names, fake_obs_data):
    # This function tests that the sampling proportions are working as expected
    artifact = mocker.MagicMock()
    artifact.load.return_value = last_names
    # Subset last names data to match fake_obs_data
    last_names = last_names.reset_index()[["name", "White", "Latino", "Black", "Asian"]]

    # Get proportion of given names for tests
    proportions = (
        pd.melt(
            last_names,
            id_vars="name",
            value_vars=["White", "Latino", "Black", "Asian"],
            var_name="race_ethnicity",
            value_name="freq",
        )
        .groupby(["race_ethnicity", "name"])["freq"]
        .sum()
    )

    # Get name frequencies for comparison
    last_names_map = get_last_name_map(
        "last_name_id",
        fake_obs_data,
        artifact,
        randomness,
        "1234",
    )
    last_name_proportions = get_name_frequency_proportions(
        last_names_map["last_name"],
        fake_obs_data["fake_observer"],
        ["race_ethnicity"],
    )

    assert np.isclose(
        last_name_proportions.sort_index(), proportions.sort_index(), atol=1e-02
    ).all()
    assert not (last_names_map["last_name"].isnull().any())

    other_seed_last_names_map = get_last_name_map(
        "last_name_id",
        fake_obs_data,
        artifact,
        randomness,
        "2345",
    )

    assert (last_names_map["last_name"] != other_seed_last_names_map["last_name"]).any()


def test_last_name_from_oldest_member(mocker):
    # This tests logic that we are sampling last_name_id for oldest member with that id.
    # Each household has an oldest member of a difference race ethnicity.
    household_1 = pd.DataFrame(
        {
            "race_ethnicity": ["Asian", "Black", "Latino"],
            "date_of_birth": [
                pd.Timestamp("1965-01-01 00:00:00"),
                pd.Timestamp("1975-01-01 00:00:00"),
                pd.Timestamp("1985-01-01 00:00:00"),
            ],
            "last_name_id": [1, 1, 1],
        }
    )
    household_2 = pd.DataFrame(
        {
            "race_ethnicity": ["Asian", "Black", "Latino"],
            "date_of_birth": [
                pd.Timestamp("1975-01-01 00:00:00"),
                pd.Timestamp("1965-01-01 00:00:00"),
                pd.Timestamp("1985-01-01 00:00:00"),
            ],
            "last_name_id": [2, 2, 2],
        }
    )
    household_3 = pd.DataFrame(
        {
            "race_ethnicity": ["Asian", "Black", "Latino"],
            "date_of_birth": [
                pd.Timestamp("1985-01-01 00:00:00"),
                pd.Timestamp("1975-01-01 00:00:00"),
                pd.Timestamp("1965-01-01 00:00:00"),
            ],
            "last_name_id": [3, 3, 3],
        }
    )
    households = pd.concat([household_1, household_2, household_3])
    fake_obs_data = {"last_name_faker": households}

    # Make fake artifact data
    last_names = pd.DataFrame(
        {
            "name": ["Name A", "Name B", "Name C"],
            "Asian": [1.0, 0.0, 0.0],
            "Black": [0.0, 1.0, 0.0],
            "Latino": [0.0, 0.0, 1.0],
        }
    )
    cols = list(last_names.columns)
    last_names = last_names.set_index(cols)

    # Mock artifact
    artifact = mocker.MagicMock()
    artifact.load.return_value = last_names

    # Map last names
    last_names_map = get_last_name_map(
        "last_name_id",
        fake_obs_data,
        artifact,
        randomness,
        "1234",
    )
    expected = pd.Series(data=["Name A", "Name B", "Name C"], index=[1, 2, 3])

    assert (last_names_map["last_name"] == expected).all()


@pytest.mark.parametrize(
    "input_address_col, street_number_col, street_name_col, unit_number_col",
    [
        (
            "address_id",
            "street_number",
            "street_name",
            "unit_number",
        ),
        (
            "employer_address_id",
            "employer_street_number",
            "employer_street_name",
            "employer_unit_number",
        ),
    ],
)
def test_address_mapping(
    mocker, input_address_col, street_number_col, street_name_col, unit_number_col
):
    # Fake synthetic pii address data
    synthetic_address_data = pd.DataFrame(
        {
            "StreetNumber": list(range(26)),
            "StreetName": list(string.ascii_lowercase),
            "Unit": list(range(26)),
        }
    )
    synthetic_address_data.set_index(["StreetNumber", "StreetName", "Unit"], inplace=True)
    # Mock artifact
    artifact = mocker.MagicMock()
    artifact.load.return_value = synthetic_address_data

    # Address_ids will just be an series of ids so we just need a unique series in a one column dataframe
    address_ids = pd.DataFrame()
    address_ids[input_address_col] = list(range(10))
    fake_obs_data = address_ids

    address_map = get_street_details_map(
        input_address_col,
        fake_obs_data,
        artifact,
        randomness,
        "1234",
    )
    expected_keys = [street_number_col, street_name_col, unit_number_col]

    assert all(street_key in expected_keys for street_key in address_map.keys())
    for street_key, series in address_map.items():
        assert (address_map[street_key].index == address_ids[input_address_col]).all()
        assert len(address_map[street_key].index.unique()) == len(
            address_map[street_key].index
        )
        assert not (address_map[street_name_col].isnull().any())

    other_seed_address_map = get_street_details_map(
        input_address_col,
        fake_obs_data,
        artifact,
        randomness,
        "2345",
    )

    for column in address_map:
        assert (address_map[column] != other_seed_address_map[column]).any()


@pytest.mark.parametrize(
    "input_address_col, zipcode_col, state_id_col, puma_col",
    [
        (
            "address_id",
            "zipcode",
            "state_id",
            "puma",
        ),
        (
            "employer_address_id",
            "employer_zipcode",
            "employer_state_id",
            "employer_puma",
        ),
    ],
)
def test_zipcode_mapping(input_address_col, zipcode_col, state_id_col, puma_col):
    """Tests ZIP code mapping logic.

    Specifically:
    - all address_ids from input should be in output map, and only once
    - For n addresses with state/puma combinations that zip codes match
    expected proportions, n needing to be sufficiently large

    This test relies on one of the few cases of two possible ZIP codes:

    state,puma,zipcode,proportion
    6,3756,90706,0.5884
    6,3756,90723,0.4116

    """
    num_ids = 1000  # the number of simulants
    num_unique_ids = int(num_ids / 2)  # the number of unique address_ids
    expected_proportion_90706 = 0.5884  # from PUMA_TO_ZIP_DATA_PATH
    expected_proportion_90723 = 0.4116  # from PUMA_TO_ZIP_DATA_PATH

    simulation_addresses = pd.DataFrame()
    simulation_addresses[input_address_col] = [f"123_{n}" for n in range(num_unique_ids)]
    simulation_addresses[state_id_col] = 6  # from PUMA_TO_ZIP_DATA_PATH
    simulation_addresses[puma_col] = 3756  # from PUMA_TO_ZIP_DATA_PATH
    simulation_addresses["silly_column"] = "yada yada yada"
    fake_obs_data = pd.concat([simulation_addresses, simulation_addresses])
    # The second level functions need to have NO duplicates which is handled in the top level of get_address_id map.
    fake_obs_data = fake_obs_data.drop_duplicates(subset=input_address_col)
    # Function under test
    mapper = get_zipcode_map(input_address_col, fake_obs_data, randomness, "1234")
    other_seed_mapper = get_zipcode_map(input_address_col, fake_obs_data, randomness, "2345")

    # Assert that each address_id is in the index once
    assert (
        len(mapper[zipcode_col].reset_index()[input_address_col].drop_duplicates())
        == num_unique_ids
    )

    # Assert that the `num_ids` simulants get assigned the correct proportion of zip code
    fake_obs_data[zipcode_col] = simulation_addresses[input_address_col].map(
        mapper[zipcode_col]
    )
    assert np.isclose(
        (fake_obs_data[zipcode_col].value_counts()[90723] / len(fake_obs_data[zipcode_col])),
        expected_proportion_90723,
        rtol=0.1,
    )
    assert np.isclose(
        (fake_obs_data[zipcode_col].value_counts()[90706] / len(fake_obs_data[zipcode_col])),
        expected_proportion_90706,
        rtol=0.1,
    )
    assert (mapper[zipcode_col] != other_seed_mapper[zipcode_col]).any()


@pytest.mark.parametrize(
    "input_address_col, city_col, state_col",
    [
        (
            "address_id",
            "city",
            "state",
        ),
        (
            "employer_address_id",
            "employer_city",
            "employer_state",
        ),
    ],
)
def test_city_address(mocker, input_address_col, city_col, state_col):
    # This tests that we assign cities based on the correct state

    # Mock artifact data
    addresses = pd.DataFrame(
        {
            "Municipality": ["San Diego", "Irvine", "Seattle", "Portland"],
            "Province": ["ca", "ca", "wa", "or"],
        }
    )
    addresses = addresses.set_index(["Municipality", "Province"])
    artifact = mocker.MagicMock()
    artifact.load.return_value = addresses

    # Fake observer data
    fake_obs_data = pd.DataFrame(
        {input_address_col: list(range(15)), state_col: ["CA", "OR", "WA"] * 5}
    )

    city_map = get_city_map(input_address_col, fake_obs_data, artifact, randomness, "1234")

    expected_keys = [city_col]
    assert all(address_key in expected_keys for address_key in city_map.keys())
    assert not (city_map[city_col].isnull().any())

    # Helper indexes for city_map
    ca_idx = fake_obs_data.index[fake_obs_data[state_col] == "CA"]
    or_idx = fake_obs_data.index[fake_obs_data[state_col] == "OR"]
    wa_idx = fake_obs_data.index[fake_obs_data[state_col] == "WA"]
    assert (city_map[city_col].loc[ca_idx].isin(["San Diego", "Irvine"])).all()
    assert (city_map[city_col].loc[or_idx] == "Portland").all()
    assert (city_map[city_col].loc[wa_idx] == "Seattle").all()

    other_seed_city_map = get_city_map(
        input_address_col, fake_obs_data, artifact, randomness, "2345"
    )

    assert (city_map[city_col] != other_seed_city_map[city_col]).any()


def test_employer_name_map(mocker, monkeypatch):
    num_known_employers = len(data_values.KNOWN_EMPLOYERS)
    known_employer_ids = list(range(num_known_employers))
    unknown_employer_ids = np.random.randint(num_known_employers, 1000, size=200)

    fake_obs_data = {
        "fake_observer": pd.DataFrame(
            {"employer_id": known_employer_ids + unknown_employer_ids.tolist()}
        )
    }
    # change odd ids so that some are the same and some are different while
    # avoiding collisions
    other_unknown_employer_ids = unknown_employer_ids.copy()
    odd_mask = (other_unknown_employer_ids % 2) == 1
    other_unknown_employer_ids[odd_mask] = (
        other_unknown_employer_ids[odd_mask] % (1000 - num_known_employers)
    ) + num_known_employers

    other_fake_obs_data = {
        "fake_observer": pd.DataFrame(
            {"employer_id": known_employer_ids + other_unknown_employer_ids.tolist()}
        )
    }
    # change odd ids

    artifact = mocker.MagicMock()
    artifact_names = [
        "".join(np.random.choice(list(string.ascii_letters), size=10).tolist())
        for _ in range(1000)
    ]
    artifact_names = pd.Series(artifact_names)

    def mock_load(data_key: str):
        return {data_keys.SYNTHETIC_DATA.BUSINESS_NAMES: artifact_names.copy()}[data_key]

    monkeypatch.setattr(artifact, "load", mock_load)

    employer_names_map = get_employer_name_map(
        "employer_id", fake_obs_data, artifact, randomness, "1234"
    )
    other_seed_employer_names_map = get_employer_name_map(
        "employer_id", fake_obs_data, artifact, randomness, "2345"
    )

    other_observer_employer_names_map = get_employer_name_map(
        "employer_id", other_fake_obs_data, artifact, randomness, "1234"
    )

    employer_names = employer_names_map["employer_name"]
    assert len(fake_obs_data["fake_observer"].drop_duplicates()) == len(employer_names)
    assert not (employer_names.isnull().any())
    assert (employer_names == other_seed_employer_names_map["employer_name"]).all()

    assert (
        employer_names.loc[employer_names.index[list(range(num_known_employers))]]
        == pd.Series([employer.employer_name for employer in data_values.KNOWN_EMPLOYERS])
    ).all()

    other_observer_employer_names = other_observer_employer_names_map["employer_name"]
    overlapping_idx = employer_names.index.intersection(other_observer_employer_names.index)
    # For different input observers the names are the same when the same employer id is provided
    assert (
        employer_names.loc[overlapping_idx]
        == other_observer_employer_names.loc[overlapping_idx]
    ).all()
    # But not all employer ids are the same, so different names are generated overall
    assert (employer_names.values != other_observer_employer_names.values).any()


def test_ssn_generation_mapping():
    """Tests randomly-generated SSN uniqueness and ranges"""
    num_unique_ids = 30_000
    simulants = pd.DataFrame()
    simulants["simulant_id"] = [f"123_{n}" for n in range(num_unique_ids)]
    simulants["has_ssn"] = [True if n % 2 == 0 else False for n in range(num_unique_ids)]

    ssn_map = pd.Series("", index=simulants["simulant_id"])
    generated_ssns = generate_ssns(
        simulants["has_ssn"].sum(), "test_ssn_generation", randomness
    )
    ssn_map[simulants.set_index("simulant_id")["has_ssn"]] = generated_ssns

    # Check that all the SSNs are unique (Half of population size plus one for no SSN)
    assert ssn_map.nunique() == len(simulants) / 2 + 1

    # Check that all SSNs are populated if ssn is True, not if False
    simulants_indexed = simulants.set_index(["simulant_id"])
    assert ssn_map[simulants_indexed["has_ssn"]].nunique() == len(simulants) // 2
    assert (ssn_map[~simulants_indexed["has_ssn"]] == "").all()

    # Check that area, group, and serial segments are within bounds
    areas = ssn_map[simulants_indexed["has_ssn"]].apply(lambda x: int(x.split("-")[0]))
    groups = ssn_map[simulants_indexed["has_ssn"]].apply(lambda x: int(x.split("-")[1]))
    serials = ssn_map[simulants_indexed["has_ssn"]].apply(lambda x: int(x.split("-")[2]))
    assert (areas != 666).all()
    assert (areas >= 1).all() and (areas <= 899).all()
    assert (groups >= 1).all() and (groups <= 99).all()
    assert (serials >= 1).all() and (serials <= 9999).all()


def test_mailing_address(mocker):
    # Mock necessary artifact and observer data
    addresses = pd.DataFrame(
        {
            "Municipality": [
                "San Diego",
                "Irvine",
                "Seattle",
                "Vancouver",
                "Portland",
                "Medford",
            ],
            "Province": ["ca", "ca", "wa", "wa", "or", "or"],
        }
    )
    addresses = addresses.set_index(["Municipality", "Province"])
    artifact = mocker.MagicMock()
    artifact.load.return_value = addresses

    # Fake observer data
    fake_obs_data = pd.DataFrame(
        {
            "address_id": list(range(150)),
            "state": ["CA", "OR", "WA"] * 50,
            "state_id": [6, 41, 53] * 50,  # State ids for CA, OR, and WA
            "puma": [3756, 1324, 11900] * 50,
            "po_box": [0, 0, 0, 1, 11, 111, 0, 0, 0, 0, 0, 0, 0, 0, 1111] * 10,
        }
    )
    fake_obs_data = fake_obs_data.set_index("address_id")
    po_box_mask = fake_obs_data["po_box"] != 0

    # Generate fake maps
    fake_maps = {}
    fake_maps["street_number"] = pd.Series(list(range(150)), index=fake_obs_data.index)
    fake_maps["street_name"] = pd.Series(["A", "B", "C"] * 50, index=fake_obs_data.index)
    fake_maps["unit_number"] = pd.Series(list(range(150)), index=fake_obs_data.index)
    fake_maps["city"] = pd.Series(
        ["San Diego", "Portland", "Seattle"] * 50, index=fake_obs_data.index
    )
    fake_maps["zipcode"] = pd.Series([92630, 97221, 98368] * 50, index=fake_obs_data.index)

    mailing_map = get_mailing_address_map(
        "address_id",
        fake_obs_data,
        artifact,
        randomness,
        fake_maps,
        "1234",
    )

    other_seed_mailing_map = get_mailing_address_map(
        "address_id",
        fake_obs_data,
        artifact,
        randomness,
        fake_maps,
        "2345",
    )
    expected_keys = [
        "mailing_address_street_number",
        "mailing_address_street_name",
        "mailing_address_unit_number",
        "mailing_address_po_box",
        "mailing_address_state",
        "mailing_address_city",
        "mailing_address_zipcode",
    ]

    assert all(street_key in expected_keys for street_key in mailing_map.keys())

    for column in ["street_number", "street_name", "unit_number"]:
        column_name = f"mailing_address_{column}"
        assert (mailing_map[column_name][po_box_mask] == "").all()
        assert (
            mailing_map[column_name][~po_box_mask] == fake_maps[column][~po_box_mask]
        ).all()

    for column in ["po_box", "state"]:
        assert (mailing_map[f"mailing_address_{column}"] == fake_obs_data[column]).all()

    for column in ["city", "zipcode"]:
        column_name = f"mailing_address_{column}"
        assert (mailing_map[column_name][po_box_mask] != fake_maps[column][po_box_mask]).any()
        assert (mailing_map[column_name] != other_seed_mailing_map[column_name]).any()


@pytest.mark.parametrize(
    "hdf_key, num_need_ids, random_seed",
    [
        (
            "/synthetic_data/ssns",
            1_000_000,
            "300",
        ),
        (
            "/synthetic_data/ssns",
            1_000_000,
            "",  # All seeds
        ),
        (
            "/synthetic_data/itins",
            100_000,
            "300",
        ),
        (
            "/synthetic_data/itins",
            100_000,
            "",  # All seeds
        ),
    ],
)
def test_id_uniqueness(artifact, hdf_key, num_need_ids, random_seed):
    """Ensure that SSNs and ITINs are unique after assigning from artifact"""
    all_seeds = [str(seed) for seed in range(1, 301)]
    try:
        ids = _load_ids(artifact, hdf_key, num_need_ids, random_seed, all_seeds)
    except FileNotFoundError:  # Allows it to be skipped for CI
        pytest.mark.skip(reason=f"Cannot find artifact at {artifact.path}")
    else:
        assert len(np.unique(ids)) == num_need_ids
