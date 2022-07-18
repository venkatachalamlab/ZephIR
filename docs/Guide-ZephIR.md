### Guide for using ZephIR

ZephIR can be used for a variety of applications and in a variety of ways. Use the example workflows as guidelines to create an ideal workflow for a particular application.

You can also watch our [video tutorial](https://youtu.be/4O9aIftvoqM) going over the workflow with an example dataset.

## General

A general workflow for using ZephIR to track keypoints in a new video.

1. Preprocess your data as needed. See [this guide](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/Guide-data.md) for tips.

2. Edit /utils/getters.py to fit IO needs for your data. See [this guide](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/Guide-data.md) for details. 

3. Navigate to the dataset directory.

4. Run: `recommend_frames --dataset=. [--options]` to determine optimal median frames to annotate and use as reference frames. Run: `recommend_frames -h` to see all CLI options.

5. Launch the annotator GUI to annotate new reference frames. See [this doc](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/annotatorGUI.md) for details on the annotator and follow [this guide](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/Guide-annotatorGUI.md) for using the annotator. 
  > If you already have annotations created with another software, prepare your annotations and edit /utils/getters.py to make sure ZephIR can load your annotations properly.

6. Run: `zephir --dataset=. [--options]` to track your keypoints. See [this guide](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/Guide-parameters.md) for details on all CLI options and [this doc](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/examples.md) for examples on how the parameters might affect your results.
  > *TIP*: If your dataset is particularly challenging and long, you can first test and optimize your parameter choices on a small subset of frames. You can either do this by editing `shape_t` your metadata to limit the scope, or using the CLI option `t_ignore` to exclude ranges of frames.

7. Check your results with annotated.mp4, which is a movie of the tracked data overlayed with the tracking results. 

8. Launch the annotator GUI for more detailed verification of the results. Follow [this guide](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/Guide-annotatorGUI.md) to provide further partial annotations as necessary.

9. Fix remaining tracking errors by either running ZephIR on individual frames with new partial annotations directly from the annotator GUI, or running ZephIR on all frames again with the new annotations (see step 6).

10. Iterate steps 7-9 as necessary.

10. Extract any downstream information. For extracting fluorescence activity, run: `extract_traces --dataset=. [--options]` to get activity traces over all frames. Run: `extract_traces -h` to see all CLI options.


## Working with subsets of keypoints

A workflow for iterating with ZephIR over multiple subsets. 
This may be helpful when certain regions in the image or groups of keypoints exhibit significantly disparate behaviors, such that each group requires a different set of parameters in order to achieve good tracking quality.


1. Preprocess your data as needed. See [this guide](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/Guide-data.md) for tips.

2. Edit /utils/getters.py to fit IO needs for your data. See [this guide](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/Guide-data.md) for details. 

3. Navigate to the dataset directory.

4. Run: `recommend_frames --dataset=. [--options]` to determine optimal median frames to annotate and use as reference frames. Run: `recommend_frames -h` to see all CLI options.

5. Launch the annotator GUI to annotate new reference frames. See [this doc](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/annotatorGUI.md) for details on the annotator and follow [this guide](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/Guide-annotatorGUI.md) for using the annotator. 
  > If you already have annotations created with another software, prepare your annotations and edit /utils/getters.py to make sure Zephir can load your annotations properly.

6. To split your keypoints into subsets to track separately, note the track number of each keypoint as shown in the "Tracks" panel of the annotator. Alternatively, you can also mark annotations with a separate provenance for each subset (i.e. `ANTT1` and `ANTT2`). Easiest way to do so is by using the macro `change_provenance`.

7. Run: `zephir --dataset=. [--options]` to track all keypoints with some initial parameters. 
See [this guide](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/Guide-parameters.md) for details on all CLI options and [this doc](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/examples.md) for examples on how the parameters might affect your results.

8. Launch the annotator GUI and find a frame where the different subsets exhibit divergent behaviors. 

9. Use the macro `overwrite_zeir_checkpoint` to change either `wlid_ref` or `exclusive_prov` (according to how you split the subsets in step 6) to specify a subset of keypoints.

10. Use the macro `overwrite_zeir_checkpoint` to adjust tracking parameters. Run `update_frame` to see how the new parameters affect tracking quality for this frame. Note a set of parameters that work well for this subset.

11. Run: `zephir --dataset=. --wlid_ref="0,1,2" --exclusive_prov=ANTT1 [--options]` to track a subset of keypoints with the optimized parameters. Use the option `wlid_ref` to specify the track numbers of the keypoints in the subset as noted in step 6, or use the option `exclusive_prov` to specify the provenance used for the subset in step 6. 
  > *WARNING* Don't forget to specify `--include_all=True` to prevent losing any annotations or results for other keypoints.

12. Check your results with `annotated.mp4`, which is a movie of the tracked data overlayed with the tracking results. 

13. Repeat steps 8-12 for each subset.

14. Launch the annotator GUI for more detailed verification of the results. Follow [this guide](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/Guide-annotatorGUI.md) to fix remaining tracking errors as necessary.

15. Extract any downstream information. For extracting fluorescence activity, run: `extract_traces --dataset=. [--options]` to get activity traces over all frames. Run: `extract_traces -h` to see all CLI options.


## Working with DeepLabCut

A workflow for using ZephIR to augment training data for DeepLabCut.

1. Preprocess your data as needed. See [this guide](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/Guide-data.md) for tips.

2. Edit /utils/getters.py to fit IO needs for your data. See [this guide](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/Guide-data.md) for details. 

3. Navigate to the dataset directory.

4. Run: `recommend_frames --dataset=. [--options]` to determine optimal median frames to annotate and use as reference frames. Run: `recommend_frames -h` to see all CLI options.

5. Launch the annotator GUI to annotate new reference frames. See [this doc](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/annotatorGUI.md) for details on the annotator and follow [this guide](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/Guide-annotatorGUI.md) for using the annotator. 

6. Rename worldlines by editing boxes under the "Tracks" panel. Each track number can be associated with a name, saved under `worldlines.h5`. We recommend linking worldlines with bodypart names here.

7. Run: `zephir --dataset=. [--options]` to track your keypoints. See [this guide](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/Guide-parameters.md) for details on all CLI options and [this doc](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/examples.md) for examples on how the parameters might affect your results.
  > *TIP*: If your dataset is particularly challenging and long, you can first test and optimize your parameter choices on a small subset of frames. You can either do this by editing `shape_t` your metadata to limit the scope, or using the CLI option `t_ignore` to exclude ranges of frames.

8. Check your results with annotated.mp4 (or with the annotator GUI), which is a movie of the tracked data overlayed with the tracking results. Note the frames with good tracking results that you would like to use for training DeepLabCut.

9. Follow [this notebook](https://github.com/venkatachalamlab/ZephIR/blob/main/notebooks/export_to_deeplabcut.ipynb) to export the annotations and results from ZephIR to a DeepLabCut format. Edit variables `t_list` with the frames you chose in step 8, and `bodyparts` with the names of the bodyparts you annotated.

10. Follow Step A and B from [this guide](https://deeplabcut.github.io/DeepLabCut/docs/standardDeepLabCut_UserGuide.html) to create and configurate a new DeepLabCut project. In particular, edit the `config.yaml` file with the `bodyparts` from steps 6 and 9. 

12. Follow [this guide](https://deeplabcut.github.io/DeepLabCut/docs/standardDeepLabCut_UserGuide.html) to train and run DeepLabCut. Start from Step F and create a new training dataset from the labeled dataset directory created in step 9.
