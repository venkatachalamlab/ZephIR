## Guide for prepping data and using getters

Use this manual for preprocessing your data and using the IO functions in getters.py.

### Preparing your data

To get the most out of ZephIR, it is recommended that you preprocess your data:

1. Denoising.

2. Global translation & rotation to fix a center and orientation of your data. In particular, fixing an orientation (i.e. worm is always facing left) will allow better use of the spring network ($\mathcal{L}$<sub>N</sub>).

3. Debleaching. If your data consists of fluorescent cells that bleach out over time, it may help to adjust the brightnesses of later frames.

4. Organize all files pertaining to a single dataset into a separate directory. ZephIR is called on dataset directories, not individual dataset files.

### getters.py

We recognize that different users will have different data structures and IO methods that they are already used to. Instead of asking users to port their datasets to mimic ours, we consolidate key IO functions such that their *output* is compatible with ZephIR, but everything else can be changed as necessary to interface with different data structures. These functions are collected in `zephir/utils/getters.py`. 

Users can either edit this file directly, or copy the file into the dataset directory and then edit the copy. The latter method is useful when a different IO process is required only for a specific dataset. ZephIR will prioritize the getters defined in the directory before defaulting to those defined in `zephir/utils/getters.py`.

Users should edit the following three functions to fit their particular use case:

&nbsp;

1. `get_slice(dataset: Path, t: int) -> np.ndarray`

Given a `pathlib.Path` to the dataset directory and an integer index t, this function should return a Numpy array containing data at a corresponding time point. The output must be 4-D, even when the dataset is single-channel or 2-D images, with dimensions ordered as: (C, Z, Y, X). This function is called and cached in `get_data`.

  > *TIP:* If your dataset has non-integer timestamps (i.e. in seconds), you can either 1) preprocess and pad your data such that subsequent frames are a fixed time interval apart and re-indexed in order, or 2) extract frames at a specified time using an ordered list of timestamps.

&nbsp;

2. `get_annotation_df(dataset: Path) -> pd.DataFrame`

Given a `pathlib.Path` to the dataset directory, this function should return a Pandas dataframe containing all existing annotations, manually created or otherwise. If the annotations were made with the provided annotator GUI, the default function will suffice. The dataframe must contain the following named columns with each row representing a single annotation: 

- t_idx: time index of each annotation
- x: x-coordinate as a float between (0, 1)
- y: y-coordinate as a float between (0, 1)
- z: z-coordinate as a float between (0, 1)
- worldline_id: track or worldline ID as an integer, where the same ID will be recognized as annotations for the same keypoint
- provenance: scorer or creator of the annotation as a byte string, where `b'ZEIR'` will be recognized as ZephIR results

  > *TIP*: Note that ZephIR will *always* save its results to either `annotations.h5` (if `save_mode=o`) or `coordinates.h5` (if `save_mode=w`), regardless of this function. If you are working iteratively or using the provided annotator GUI for verifying results, revert this function back to the default after the initial pass.

&nbsp;

3. `get_metadata(dataset: Path) -> dict`

Given a `pathlib.Path` to the dataset directory, this function should return a Python dictionary containing key metadata for the dataset. The output dictionary must contain the following keys and associated values:

- shape_t: number of frames
- shape_c: number of channels
- shape_z: size of the data in the z-axis (=1 if 2-D)
- shape_y: size of the data in the y-axis
- shape_x: size of the data in the x-axis

  > *TIP*: You can either create a json or equivalent file with those values in the dataset directory, or you can even hard-code them here for your dataset. Note that `shape_t` need not be accurate - ZephIR will only analyze up to the first `shape_t` frames in the data. Adjusting this value in the metadata effectively adjusts the scope of the analysis. For all other keys, inaccurate values may cause errors!

