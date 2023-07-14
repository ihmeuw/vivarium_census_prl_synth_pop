from bdb import BdbQuit
import functools
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from loguru import logger

from vivarium_census_prl_synth_pop.constants import paths
from vivarium_census_prl_synth_pop.utilities import build_output_dir


def handle_exceptions(func: Callable, with_debugger: bool) -> Callable:
    """Drops a user into an interactive debugger if func raises an error."""
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (BdbQuit, KeyboardInterrupt):
            raise
        except Exception as e:
            logger.exception("Uncaught exception {}".format(e))
            if with_debugger:
                import pdb
                import traceback

                traceback.print_exc()
                pdb.post_mortem()
            raise

    return wrapped


def validate_args(
    mark_best: bool, test_run: bool, label_version: Optional[str] = None
) -> None:
    if mark_best and test_run:
        raise RuntimeError(
            "A test run can't be marked best. "
            "Please remove either the mark best or the test run flag."
        )
    if label_version is not None:
        expected_version_format = re.compile("^\d+\.\d+\.\d+$")
        if expected_version_format.match(label_version):
            pass
        else:
            raise ValueError(
                f"Version '{label_version}' is not of correct format. "
                "Format for version should be '#.#.#'"
            )


def build_final_results_directory(
    results_dir: str,
    version: Optional[str] = None,
) -> Tuple[Path, Path]:
    tmp = paths.FINAL_RESULTS_DIR_NAME / datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if version is None:
        subdir = tmp / "pseudopeople_input_data_usa"
    else:
        subdir = tmp / f"pseudopeople_input_data_usa_{version}"
    final_output_dir = build_output_dir(
        Path(results_dir),
        subdir=subdir,
    )
    raw_output_dir = Path(results_dir) / paths.RAW_RESULTS_DIR_NAME

    return raw_output_dir, final_output_dir


def finish_post_processing(final_output_dir: Path, test_run: bool, mark_best: bool) -> None:
    """Every run of `make_results` should
    1. Copy the CHANGELOG to the output directory
    2. Update symlinks if successful
    """

    try:
        shutil.copyfile(paths.REPO_DIR / "CHANGELOG.rst", final_output_dir / "CHANGELOG.rst")
    except:
        raise RuntimeError("Unable to copy the CHANGELOG.rst. Did not generate new symlinks.")

    if test_run:
        logger.info("Test run - not marking results as latest.")
    else:
        _create_results_link(final_output_dir, paths.LATEST_DIR_NAME)

    if mark_best:
        _create_results_link(final_output_dir, paths.BEST_DIR_NAME)


def _create_results_link(output_dir: Path, link_name: Path) -> None:
    logger.info(f"Marking results as '{link_name}': {str(output_dir)}.")
    # Create simulation directory link (two levels higher than output_dir)
    # Create relative link

    # output_root_dir = output_dir.parent
    # link_dir = output_root_dir / link_name
    link_dir = (
        Path(str(output_dir).split(str(paths.FINAL_RESULTS_DIR_NAME))[0])
        / paths.FINAL_RESULTS_DIR_NAME
        / link_name
    )
    link_dir.unlink(missing_ok=True)
    link_dir.symlink_to(output_dir, target_is_directory=True)
