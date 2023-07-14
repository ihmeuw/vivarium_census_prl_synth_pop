import shutil
import sys
import uuid
from pathlib import Path

from jobmon.client.tool import Tool
from loguru import logger

from vivarium_census_prl_synth_pop.utilities import (
    build_output_dir,
    get_all_simulation_seeds,
)


def run_make_results_workflow(
    raw_output_dir: Path,
    final_output_dir: Path,
    extension: str,
    public_sample: bool,
    seed: str,
    artifact_path: str,
    queue: str,
    peak_memory: int,
    max_runtime: str,
    verbose: int,
) -> str:
    """Creates and runs a jobmon workflow to build results datasets
    from the raw data output by observers
    """
    logger.info(
        "Starting jobmon 'make_results' workflow. Results will be written to "
        f"{final_output_dir} with output format {extension}."
    )
    wf_uuid = uuid.uuid4()

    # Format arguments to be cli-friendly
    extension_arg = f"--extension {extension}" if extension != "" else ""
    public_sample_arg = "--public-sample" if public_sample else ""
    verbose_arg = "-" + "v" * verbose if verbose else ""
    seed_arg = f"--seed {seed}" if seed != "" else ""

    # Create tool
    tool = Tool(name="vivarium_census_prl_synth_pop.make_results")

    # Create a workflow
    workflow = tool.create_workflow(name=f"make_results_workflow_{wf_uuid}")

    # Create task templates
    log_dir = build_output_dir(final_output_dir, subdir="logs")
    template_make_results = tool.get_task_template(
        template_name="make_results_template",
        default_compute_resources={
            "queue": queue,
            "cores": 1,
            "memory": peak_memory,
            "runtime": max_runtime,
            "project": "proj_simscience_prod",
            "stdout": str(log_dir),
            "stderr": str(log_dir),
        },
        default_cluster_name="slurm",
        # Build the cli command to be run. The spaces after each part or required
        command_template=(
            f"{shutil.which('jobmon_make_results_runner')} "
            "{raw_output_dir} "
            "{final_output_dir} "
            "{verbose} "
            "{extension} "
            "{public_sample} "
            "{seed} "
            "--artifact-path {artifact_path} "
        ),
        node_args=["seed"],
        task_args=[
            "raw_output_dir",
            "final_output_dir",
            "extension",
            "public_sample",
            "artifact_path",
        ],
        op_args=[
            "verbose",
        ],
    )

    # Create tasks
    if seed_arg == "":  # Process all seeds in parallel
        # All raw results are in the format <raw_output_dir>/<observer>/<observer>_<seed>.csv.bz2
        seeds = get_all_simulation_seeds(raw_output_dir)
        seed_args = [f"--seed {seed}" for seed in seeds]
        task_make_results = template_make_results.create_tasks(
            upstream_tasks=[],
            raw_output_dir=str(raw_output_dir),
            final_output_dir=str(final_output_dir),
            verbose=verbose_arg,
            extension=extension_arg,
            public_sample=public_sample_arg,
            seed=seed_args,
            artifact_path=artifact_path,
        )
        workflow.add_tasks(task_make_results)
    else:  # Process the single provided seed
        task_make_results = template_make_results.create_task(
            name="make_results_task",
            upstream_tasks=[],
            raw_output_dir=str(raw_output_dir),
            final_output_dir=str(final_output_dir),
            verbose=verbose_arg,
            extension=extension_arg,
            public_sample=public_sample_arg,
            seed=seed_arg,
            artifact_path=artifact_path,
        )
        workflow.add_task(task_make_results)

    # Run the workflow
    workflow.bind()
    logger.info(f"Running workflow with ID {workflow.workflow_id}")
    logger.info("For full information see the Jobmon GUI:")
    logger.info(
        f"https://jobmon-gui.ihme.washington.edu/#/workflow/{workflow.workflow_id}/tasks"
    )
    status = workflow.run(configure_logging=True)
    logger.info(f"Workflow {workflow.workflow_id} completed with status {status}.")

    return status
