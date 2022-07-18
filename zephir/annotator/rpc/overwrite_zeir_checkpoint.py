from logging import Logger
from pathlib import Path
from typing import List

from ..data import AnnotationTable, WorldlineTable
from ...methods.overwrite_checkpoint import overwrite_checkpoint

from ._utilities import default_args


@default_args("None, None")
def overwrite_zeir_checkpoint(
    dataset: Path,
    annotations: AnnotationTable,
    worldlines: WorldlineTable,
    window_state,
    arg: str,
    logger: Logger
) -> List[dict]:
	"""
	arg: key, value
		key: Name of item in checkpoint to overwrite.
		value: New item to write into checkpoint.
	ex: "kn_max", "10": Updates max number of neighbor spring connections to 10.
		"clip_grad", "0.1": Updates gradient clipping ceiling to 0.1.
	"""

	arg_list = arg.replace(" ", "").split(",")

	key = str(arg_list[0])
	value = eval(arg_list[1])

	overwrite_checkpoint(dataset, key, value)

	return [{"type": "annotations/get_annotations"}]
