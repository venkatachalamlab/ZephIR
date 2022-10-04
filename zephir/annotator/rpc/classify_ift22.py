from logging import Logger
from pathlib import Path
from typing import List

from ..data import AnnotationTable, WorldlineTable
from ...classifier.inference import perform_inference

from ._utilities import default_args


@default_args("_now")
def classify_ift22(
    dataset: Path,
    annotations: AnnotationTable,
    worldlines: WorldlineTable,
    window_state,
    arg: str,
    logger: Logger
) -> List[dict]:
    """
    arg: time,
        time: Time to run the NeuronClassifier on.
    """

    arg_list = arg.replace(" ", "").split(",")

    if arg_list[0] == '_now':
        t_idx = window_state['t_idx']
    else:
        t_idx = int(arg_list[0])

    ret = perform_inference(
        dataset=Path(dataset),
        annotations_df=annotations.df,
        worldlines_df=worldlines.df,
        t_idx=t_idx,
    )
    if ret is None:
        return [{
            "type": "annotations/get_annotations"
        }, {
            "type": "worldlines/get_worldlines"
        }]
    else:
        wlid, names = ret

    new_names = [b'null'] * len(worldlines)
    for i, w in enumerate(worldlines.df.id):
        if w in wlid:
            idx = list(wlid).index(w)
            new_names[i] = names[idx]
        else:
            new_names[i] = worldlines.get(w).name

    worldlines.df.name = new_names

    return [{
        "type": "annotations/get_annotations"
    }, {
        "type": "worldlines/get_worldlines"
    }]
