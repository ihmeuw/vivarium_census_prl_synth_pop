from pathlib import Path
from typing import Tuple

import click
from loguru import logger

from vivarium_census_prl_synth_pop.constants import metadata, paths
from vivarium_census_prl_synth_pop.tools import (
    build_artifacts,
    build_final_results_directory,
    build_results,
    configure_logging_to_terminal,
    finish_post_processing,
    handle_exceptions,
    subset_results_by_state,
    validate_args,
)
from vivarium_census_prl_synth_pop.tools.jobmon import run_make_results_workflow


@click.command()
@click.option(
    "-l",
    "--location",
    default="United States of America",
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
    main = handle_exceptions(
        func=build_artifacts, exceptions_logger=logger, with_debugger=with_debugger
    )
    main(location, output_dir, append or replace_keys, replace_keys, verbose)


@click.command()
@click.argument("output_dir", type=click.Path(exists=True))
@click.argument(
    "label_version",
    # Version for final results and metadata file. This should be of format '#.#.#'
)
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
    "--public-sample",
    is_flag=True,
    help="Generates results for the small-scale public sample data.",
)
@click.option(
    "-s",
    "--seed",
    type=str,
    default="",
    show_default=True,
    help="Provide seed in order to run only on the files corresponding to that "
    "seed. If no seed it provided, will run on all files.",
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
    "The estimated maximum memory usage in GB of an individual task. "
    "The tasks will be run with this as a limit. ",
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
    "queue. The maximum supported runtime is 3 days. Keep in mind that "
    "runtimes by node vary wildly. ",
)
def make_results(
    output_dir: str,
    label_version: str,
    verbose: int,
    with_debugger: bool,
    mark_best: bool,
    test_run: bool,
    public_sample: bool,
    seed: str,
    artifact_path: str,
    jobmon: bool,
    queue: str,
    peak_memory: int,
    max_runtime: str,
) -> None:
    """Create final results datasets from the raw results found in the OUTPUT_DIR
    and store them in a time-stamped and versioned directory structure using LABEL_VERSION.
    \f
    :param output_dir: Directory where the raw results are stored.
    :param label_version: Version for final results and metadata file. This should be of format '#.#.#'
    :param verbose: Configure logging verbosity.
    :param with_debugger: Drop into python debugger if an error occurs.
    :param mark_best: Marks this version of results with a 'best' symlink.
    :param test_run: Skips updating the 'latest' symlink with this version of results.
    :param extension: File type to write results out as. Supported file types are hdf and parquet.
    :param public_sample: Generates results for the small-scale public sample data.
    :param seed: Provide seed in order to run only on the files corresponding to that seed. If no seed it provided, will run on all files.
    :param artifact_path: Path to artifact used in simulation.
    :param jobmon: Launches a jobmon workflow to generate results.
    :param queue: NOTE: only required if --jobmon. The cluster queue to assign jobmon tasks to. long.q allows for much longer runtimes although there may be reasons to send jobs to that queue even if they don't have runtime constraints (eg node availability).
    :param peak_memory: NOTE: only required if --jobmon. The estimated maximum memory usage in GB of an individual task. The tasks will be run with this as a limit.
    :param max_runtime: NOTE: only required if --jobmon. The estimated maximum runtime ('hh:mm:ss') of the jobmon tasks. Once this time limit is hit, the cluster will terminate jobs regardless of queue. The maximum supported runtime is 3 days. Keep in mind that runtimes by node vary wildly.

    """
    configure_logging_to_terminal(verbose)
    validate_args(mark_best=mark_best, test_run=test_run, label_version=label_version)
    raw_output_dir, final_output_dir = build_final_results_directory(
        output_dir, label_version
    )
    logger.info(f"Final results directory: {str(final_output_dir)}")
    cluster_requests = {
        "queue": queue,
        "peak_memory": peak_memory,
        "max_runtime": max_runtime,
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
        kwargs["verbose"] = verbose
    else:  # run from current node
        func = build_results
        kwargs = {}
    main = handle_exceptions(func=func, exceptions_logger=logger, with_debugger=with_debugger)
    status = main(
        raw_output_dir,
        final_output_dir,
        public_sample,
        seed,
        artifact_path,
        **kwargs,
    )
    if status and status != "D":
        raise RuntimeError("Jobmon status did not finish successfully")
    finish_post_processing(final_output_dir, test_run, mark_best)


@click.command()
@click.argument("raw_output_dir", type=click.Path(exists=True))
@click.argument("final_output_dir", type=click.Path(exists=True))
@click.option("-v", "verbose", count=True, help="Configure logging verbosity.")
@click.option(
    "--public-sample",
    is_flag=True,
)
@click.option(
    "-s",
    "--seed",
    type=str,
    default="",
    show_default=True,
    help="Provide seed in order to run only on the files corresponding to that "
    "seed. If no seed it provided, will run on all files.",
)
@click.option(
    "-i", "--artifact-path", type=click.Path(exists=True), default=paths.DEFAULT_ARTIFACT
)
def jobmon_make_results_runner(
    raw_output_dir: str,
    final_output_dir: str,
    verbose: int,
    public_sample: bool,
    seed: str,
    artifact_path: str,
) -> None:
    configure_logging_to_terminal(verbose)
    main = handle_exceptions(
        func=build_results, exceptions_logger=logger, with_debugger=False
    )
    main(
        Path(raw_output_dir),
        Path(final_output_dir),
        public_sample,
        seed,
        artifact_path,
    )


@click.command()
@click.argument("processed_results_dir", type=click.Path(exists=True))
@click.option(
    "--pdb",
    "with_debugger",
    is_flag=True,
    help="Drop into python debugger if an error occurs.",
)
@click.option("-v", "verbose", count=True, help="Configure logging verbosity.")
@click.option(
    "-l",
    "--state",
    type=str,
    default="RI",
    show_default=True,
    help="State to subset process results to obtain a smaller dataset for one "
    "specific geographic location. This should be the two letter postal "
    "abbreviation.",
)
def make_state_results(
    processed_results_dir: str,
    verbose: int,
    with_debugger: bool,
    state: str,
) -> None:
    """Subset the processed results found in PROCESSED_RESULTS_DIR to a specific STATE.
    \f
    :param processed_results_dir: Directory where the already-processed results exist.
    :param verbose: Configure logging verbosity.
    :param with_debugger: Drop into python debugger if an error occurs.
    :param state: The state to subset to. This should be the two letter postal abbreviation.
    """
    resolved_results_dir = Path(processed_results_dir).resolve()
    # Ensure only a single expected results directory exists
    resolved_results_dir = [
        p
        for p in list(resolved_results_dir.glob("**"))
        if f"{paths.PROCESSED_RESULTS_DIR_NAME_BASE}" in p.name
    ]
    if len(resolved_results_dir) < 1:
        raise ValueError(
            f"The subdirectory '{paths.PROCESSED_RESULTS_DIR_NAME_BASE}_<VERSION>' "
            "is expected but does not exist at the location provided."
        )
    if len(resolved_results_dir) > 1:
        raise ValueError(
            f"Multiple subdirectories '{paths.PROCESSED_RESULTS_DIR_NAME_BASE}_<VERSION>' "
            "exist at the location provided when only one is expected."
        )
    configure_logging_to_terminal(verbose)
    main = handle_exceptions(
        func=subset_results_by_state, exceptions_logger=logger, with_debugger=with_debugger
    )
    main(resolved_results_dir[0], state)
