from typing import Tuple

import click
from loguru import logger
from vivarium.framework.utilities import handle_exceptions

from vivarium_census_prl_synth_pop.constants import metadata, paths
from vivarium_census_prl_synth_pop.tools import (
    build_artifacts,
    configure_logging_to_terminal,
)
from vivarium_census_prl_synth_pop.tools.jobmon import run_make_results_workflow
from vivarium_census_prl_synth_pop.tools.make_results import do_build_results


@click.command()
@click.option(
    "-l",
    "--location",
    default="all",
    show_default=True,
    type=click.Choice(metadata.LOCATIONS + ["all"]),
    help=(
        "Location for which to make an artifact. Note: prefer building archives on the cluster.\n"
        'If you specify location "all" you must be on a cluster node.'
    ),
)
@click.option(
    "-o",
    "--output-dir",
    default=str(paths.ARTIFACT_ROOT),
    show_default=True,
    type=click.Path(),
    help="Specify an output directory. Directory must exist.",
)
@click.option(
    "-a", "--append", is_flag=True, help="Append to the artifact instead of overwriting."
)
@click.option("-r", "--replace-keys", multiple=True, help="Specify keys to overwrite")
@click.option("-v", "verbose", count=True, help="Configure logging verbosity.")
@click.option(
    "--pdb",
    "with_debugger",
    is_flag=True,
    help="Drop into python debugger if an error occurs.",
)
def make_artifacts(
    location: str,
    output_dir: str,
    append: bool,
    replace_keys: Tuple[str, ...],
    verbose: int,
    with_debugger: bool,
) -> None:
    configure_logging_to_terminal(verbose)
    main = handle_exceptions(build_artifacts, logger, with_debugger=with_debugger)
    main(location, output_dir, append or replace_keys, replace_keys, verbose)


@click.command()
@click.argument("output_dir", type=click.Path(exists=True))
@click.option("-v", "verbose", count=True, help="Configure logging verbosity.")
@click.option(
    "--pdb",
    "with_debugger",
    is_flag=True,
    help="Drop into python debugger if an error occurs.",
)
@click.option(
    "-b",
    "--mark-best",
    is_flag=True,
    help="Marks this version of results with a 'best' symlink.",
)
@click.option(
    "-t",
    "--test-run",
    is_flag=True,
    help="Skips updating the 'latest' symlink with this version of results.",
)
@click.option(
    "-a",
    "--artifact-path",
    type=click.Path(exists=True),
    default=paths.DEFAULT_ARTIFACT,
    show_default=True,
    help="Path to artifact used in simulation.",
)
@click.option(
    "-j",
    "--jobmon",
    is_flag=True,
    help="Launches a jobmon workflow to generate results.",
)
@click.option(
    "-q",
    "--queue",
    type=click.Choice(["all.q", "long.q"]),
    help="NOTE: only required if --jobmon. "
    "The cluster queue to assign jobmon tasks to. long.q allows for much "
    "longer runtimes although there may be reasons to send jobs to that queue "
    "even if they don't have runtime constraints (eg node availability). ",
)
@click.option(
    "-m",
    "--peak-memory",
    type=int,
    help="NOTE: only required if --jobmon. "
    "The estimated maximum memory usage in GB of an individual simulate job. "
    "The simulations will be run with this as a limit. ",
)
@click.option(
    "-r",
    "--max-runtime",
    type=str,
    help="NOTE: only required if --jobmon. "
    "The estimated maximum runtime ('hh:mm:ss') of the jobmon tasks. "
    "Once this time limit is hit, the cluster will terminate jobs regardless of "
    "queue. The maximum supported runtime is 3 days. Keep in mind that the "
    "session you are launching from must be able to live at least as long "
    "as the simulation jobs, and that runtimes by node vary wildly. ",
)
@click.option(
    "-P",
    "--project",
    type=click.Choice(
        [
            "proj_simscience",
            "proj_simscience_prod",
        ]
    ),
    help="NOTE: only required if --jobmon. "
    "The cluster project under which to run the jobmon workflow. ",
)
def make_results(
    output_dir: str,
    verbose: int,
    with_debugger: bool,
    mark_best: bool,
    test_run: bool,
    artifact_path: str,
    jobmon: bool,
    queue: str,
    peak_memory: int,
    max_runtime: str,
    project: str,
) -> None:
    """Create final results datasets from the raw results output by observers"""
    configure_logging_to_terminal(verbose)
    cluster_requests = {
        "queue": queue,
        "peak_memory": peak_memory,
        "max_runtime": max_runtime,
        "project": project,
    }
    if not jobmon and (queue or peak_memory or max_runtime or project):
        requests = {k: v for k, v in cluster_requests.items() if v is not None}
        logger.info(
            "Passing in resource requests is only necessary for --jobmon. "
            f"The following were provided but will not be used: {requests}"
        )
    if jobmon and not (queue and peak_memory and max_runtime and project):
        missing_requests = [f"--{k}" for k, v in cluster_requests.items() if v is None]
        raise RuntimeError(
            "Passing in --jobmon requires all cluster request args to be defined. "
            f"Missing: {missing_requests}."
        )

    if jobmon:
        main = handle_exceptions(
            func=run_make_results_workflow, logger=logger, with_debugger=with_debugger
        )
        kwargs = cluster_requests
        main(output_dir, mark_best, test_run, artifact_path, **kwargs)
    else:  # run from current node
        main = handle_exceptions(
            func=do_build_results, logger=logger, with_debugger=with_debugger
        )
        main(output_dir, mark_best, test_run, artifact_path)
