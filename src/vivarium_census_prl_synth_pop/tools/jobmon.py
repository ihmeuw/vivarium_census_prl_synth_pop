import shutil
import uuid

from jobmon.client.tool import Tool
from loguru import logger


def run_make_results_workflow(
    output_dir: str, mark_best: bool, test_run: bool, artifact_path: str
) -> None:
    """Creates and runs a jobmon workflow to build results datasets
    from the raw data output by observers
    """
    logger.info(f"Starting make_results workflow {output_dir}")
    wf_uuid = uuid.uuid4()

    # Deal with boolean args - click either the flag or nothing, not True/False
    # TODO: there's got to be a better way to do this?
    if mark_best:
        mark_best_arg = "--mark-best"
    else:
        mark_best_arg = ""
    if test_run:
        test_run_arg = "--test-run"
    else:
        test_run_arg = ""

    # Create tool
    tool = Tool(name="vivarium_census_prl_synth_pop.make_results")

    # Create a workflow
    workflow = tool.create_workflow(name=f"make_results_workflow_{wf_uuid}")

    # Create task templates
    template_make_results = tool.get_task_template(
        template_name="make_results_template",
        default_compute_resources={
            "queue": "long.q",
            "cores": 1,
            "memory": "70G",
            "runtime": "1:00:00",
            "project": "proj_simscience",
            "stdout": str(output_dir),
            "stderr": str(output_dir),
        },
        default_cluster_name="slurm",
        command_template=(
            f"{shutil.which('build_results')} "
            "{output_dir} "
            "{mark_best} "
            "{test_run} "
            "--artifact-path {artifact_path} "
            "-vvv "
        ),
        node_args=[],
        task_args=["output_dir", "mark_best", "test_run", "artifact_path"],
        op_args=[],
    )
    # Create tasks
    task_make_results = template_make_results.create_task(
        name="make_results_task",
        upstream_tasks=[],
        output_dir=output_dir,
        mark_best=mark_best_arg,
        test_run=test_run_arg,
        artifact_path=artifact_path,
    )
    workflow.add_task(task_make_results)

    # Run the workflow
    workflow.run(configure_logging=True)
