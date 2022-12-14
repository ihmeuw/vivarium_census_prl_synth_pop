import pytest

from vivarium_census_prl_synth_pop.components import observers

def test_instantiate_base_observer():
    """Expact a TypeError when trying to instantiate an abc"""
    with pytest.raises(Exception):
        observers.BaseObserver()

