from gbd_mapping import ModelableEntity, causes, covariates, risk_factors
from vivarium.framework.artifact import EntityKey
from vivarium_inputs.mapping_extension import alternative_risk_factors


def get_entity(key: EntityKey) -> ModelableEntity:
    # Map of entity types to their gbd mappings.
    type_map = {
        "cause": causes,
        "covariate": covariates,
        "risk_factor": risk_factors,
        "alternative_risk_factor": alternative_risk_factors,
    }
    return type_map[key.type][key.name]
