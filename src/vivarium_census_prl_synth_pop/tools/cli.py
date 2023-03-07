from pathlib import Path
from typing import Tuple, Union

import click
from loguru import logger
from vivarium.framework.utilities import handle_exceptions

from vivarium_census_prl_synth_pop.constants import metadata, paths
from vivarium_census_prl_synth_pop.tools import (
    build_artifacts,
    configure_logging_to_terminal,
)
from vivarium_census_prl_synth_pop.tools.jobmon import run_make_results_workflow
from vivarium_census_prl_synth_pop.tools.make_results import build_results
from vivarium_census_prl_synth_pop.utilities import build_final_results_directory


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
    "-i",
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
    default="all.q",
    show_default=True,
    help="NOTE: only required if --jobmon. "
    "The cluster queue to assign jobmon tasks to. long.q allows for much "
    "longer runtimes although there may be reasons to send jobs to that queue "
    "even if they don't have runtime constraints (eg node availability). ",
)
@click.option(
    "-m",
    "--peak-memory",
    type=int,
    default=200,
    show_default=True,
    help="NOTE: only required if --jobmon. "
    "The estimated maximum memory usage in GB of an individual simulate job. "
    "The simulations will be run with this as a limit. ",
)
@click.option(
    "-r",
    "--max-runtime",
    type=str,
    default="4:00:00",
    show_default=True,
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
    default="proj_simscience",
    show_default=True,
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
    logger.info("Creating final results directory.")
    raw_output_dir, final_output_dir = build_final_results_directory(output_dir)
    cluster_requests = {
        "queue": queue,
        "peak_memory": peak_memory,
        "max_runtime": max_runtime,
        "project": project,
    }
    user_cluster_requests = {
        k: v
        for k, v in cluster_requests.items()
        if click.get_current_context().get_parameter_source(k).name != "DEFAULT"
    }
    if not jobmon and bool(user_cluster_requests):
        requests = {k: v for k, v in cluster_requests.items() if k in user_cluster_requests}
        logger.warning(
            "Passing in resource requests is only necessary for --jobmon. "
            f"The following were provided but will not be used: {requests}"
        )

    if jobmon:
        func = run_make_results_workflow
        kwargs = cluster_requests
    else:  # run from current node
        func = build_results
        kwargs = {}
    main = handle_exceptions(func=func, logger=logger, with_debugger=with_debugger)
    main(raw_output_dir, final_output_dir, mark_best, test_run, artifact_path, **kwargs)


@click.command()
@click.argument("raw_output_dir", type=click.Path(exists=True))
@click.argument("final_output_dir", type=click.Path(exists=True))
@click.option("-v", "verbose", count=True, help="Configure logging verbosity.")
@click.option(
    "--pdb",
    "with_debugger",
    is_flag=True,
    help="Drop into python debugger if an error occurs.",
)
@click.option("-b", "--mark-best", is_flag=True)
@click.option(
    "-t",
    "--test-run",
    is_flag=True,
)
@click.option(
    "-i", "--artifact-path", type=click.Path(exists=True), default=paths.DEFAULT_ARTIFACT
)
def jobmon_make_results_runner(
    raw_output_dir: Path,
    final_output_dir: Path,
    verbose: int,
    with_debugger: bool,
    mark_best: bool,
    test_run: bool,
    artifact_path: Union[str, Path],
) -> None:
    configure_logging_to_terminal(verbose)
    main = handle_exceptions(func=build_results, logger=logger, with_debugger=with_debugger)
    main(raw_output_dir, final_output_dir, mark_best, test_run, artifact_path)
