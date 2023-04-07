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
        "vivarium==1.0.2",
        "vivarium_public_health==0.10.22",
        "click",
        "gbd_mapping>=3.0.6, <4.0.0",
        "jinja2",
        "loguru",
        "numpy",
        "pandas>=1.0.0, <2.0.0",
        "scipy",
        "tables",
        "pyyaml",
        "faker==13.6.0",
        "pyarrow",
    ]

    # use "pip install -e .[dev]" to install required components + extra components
    data_requires = [
        "jobmon_installer_ihme==10.6.0",
        "vivarium_cluster_tools>=1.3.8",
        "vivarium_inputs[data]==4.0.10",
    ]
    test_requirements = [
        "pytest",
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
            "data": data_requires,
            "dev": test_requirements + data_requires,
        },
        zip_safe=False,
        entry_points="""
            [console_scripts]
            make_artifacts=vivarium_census_prl_synth_pop.tools.cli:make_artifacts
            make_results=vivarium_census_prl_synth_pop.tools.cli:make_results
            jobmon_make_results_runner=vivarium_census_prl_synth_pop.tools.cli:jobmon_make_results_runner
        """,
    )
