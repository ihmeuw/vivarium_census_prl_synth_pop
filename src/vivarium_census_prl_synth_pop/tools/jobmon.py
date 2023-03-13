import shutil
import uuid
from pathlib import Path
from typing import Union

from jobmon.client.tool import Tool
from loguru import logger


def run_make_results_workflow(
    raw_output_dir: Path,
    final_output_dir: Path,
    mark_best: bool,
    test_run: bool,
    seed: str,
    artifact_path: Union[str, Path],
    queue: str,
    peak_memory: int,
    max_runtime: str,
    verbose: int,
) -> None:
    """Creates and runs a jobmon workflow to build results datasets
    from the raw data output by observers
    """
    logger.info(
        "Starting jobmon 'make_results' workflow. Results will be written to "
        f"{final_output_dir}"
    )
    wf_uuid = uuid.uuid4()

    # Format arguments to be cli-friendly
    # TODO: there's got to be a better way to do this?
    mark_best_arg = "--mark-best" if mark_best else ""
    test_run_arg = "--test-run" if test_run else ""
    verbose_arg = "-" + "v" * verbose if verbose else ""
    seed_arg = f"--seed {seed}" if seed != "" else ""

    # Create tool
    tool = Tool(name="vivarium_census_prl_synth_pop.make_results")

    # Create a workflow
    workflow = tool.create_workflow(name=f"make_results_workflow_{wf_uuid}")

    # Create task templates
    template_make_results = tool.get_task_template(
        template_name="make_results_template",
        default_compute_resources={
            "queue": queue,
            "cores": 1,
            "memory": peak_memory,
            "runtime": max_runtime,
            "project": "proj_simscience_prod",
            "stdout": str(final_output_dir),
            "stderr": str(final_output_dir),
        },
        default_cluster_name="slurm",
        # Build the cli command to be run. The spaces after each part or required
        command_template=(
            f"{shutil.which('jobmon_make_results_runner')} "
            "{raw_output_dir} "
            "{final_output_dir} "
            "{verbose} "
            "{mark_best} "
            "{test_run} "
            "{seed} "
            "--artifact-path {artifact_path} "
        ),
        # node_args=["seed"],  # TODO: parameterize by seed
        task_args=[
            "raw_output_dir",
            "final_output_dir",
            "mark_best",
            "test_run",
            "seed",  # TODO: move seed to a node arg
            "artifact_path",
        ],
        op_args=[
            "verbose",
        ],
    )

    # Create tasks
    task_make_results = template_make_results.create_task(
        name="make_results_task",
        upstream_tasks=[],
        raw_output_dir=raw_output_dir,
        final_output_dir=final_output_dir,
        verbose=verbose_arg,
        mark_best=mark_best_arg,
        test_run=test_run_arg,
        seed=seed_arg,
        artifact_path=artifact_path,
    )
    workflow.add_task(task_make_results)

    # Run the workflow
    workflow.run(configure_logging=True)
