#!/usr/bin/env python
import os

from setuptools import find_packages, setup

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    src_dir = os.path.join(base_dir, "src")

    about = {}
    with open(os.path.join(src_dir, "vivarium_census_prl_synth_pop", "__about__.py")) as f:
        exec(f.read(), about)

    with open(os.path.join(base_dir, "README.rst")) as f:
        long_description = f.read()

    install_requirements = [
        "vivarium>=3.0.0 , <4.0.0",
        "vivarium_public_health>=3.0.0, <4.0.0",
        "click",
        "gbd_mapping>=3.0.6, <4.0.0",
        "jinja2",
        "loguru",
        "numpy<2.0.0",
        "pandas>=1.0.0",
        "scipy",
        "tables",
        "pyyaml",
        "faker==13.6.0",
        "pyarrow",
        "pseudopeople",
    ]

    # use "pip install -e .[dev]" to install required components + extra components
    data_requirements = ["vivarium_inputs[data]==4.1.0"]
    cluster_requirements = ["jobmon_installer_ihme==10.6.2", "vivarium_cluster_tools>=2.0.0, <3.0.0"]
    test_requirements = [
        "pytest",
        "pytest-cov",
        "pytest-mock",
    ]

    setup(
        name=about["__title__"],
        version=about["__version__"],
        description=about["__summary__"],
        long_description=long_description,
        license=about["__license__"],
        url=about["__uri__"],
        author=about["__author__"],
        author_email=about["__email__"],
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        include_package_data=True,
        install_requires=install_requirements,
        extras_require={
            "test": test_requirements,
            "cluster": cluster_requirements,
            "data": data_requirements + cluster_requirements,
            "dev": test_requirements + cluster_requirements,
        },
        zip_safe=False,
        entry_points="""
            [console_scripts]
            make_artifacts=vivarium_census_prl_synth_pop.tools.cli:make_artifacts
            make_results=vivarium_census_prl_synth_pop.tools.cli:make_results
            jobmon_make_results_runner=vivarium_census_prl_synth_pop.tools.cli:jobmon_make_results_runner
            make_state_results=vivarium_census_prl_synth_pop.tools.cli:make_state_results
        """,
    )
