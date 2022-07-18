## Tips and explanations for parameters

- `dataset`: Path to data directory to analyze.
- `load_checkpoint`: Load existing checkpoint.pt file and resume from last run. *default: False*
- `load_args`: Load parameters from existing args.json file. *default: False*
- `allow_rotation`: Enable optimizable parameter for rotating image descriptors. This may be helpful for datasets that have clear rotational changes in shape, but is generally superfluous for nucleus tracking. *default: False*
- `channel`: Choose which data channel to analyze. Leave out to use all available channels.
- `clip_grad`: Maximum value for gradients for gradient descent. Use -1 to uncap. *default: 1.0*
    > *TIP:* If motion is small, set lower to ~0.1. This is a more aggressive tactic than lr_ceiling.
- `cuda`: Toggle to allow GPU usage if a CUDA-compatible GPU is available for use. *default: True*
- `dimmer_ratio`: Coefficient for dimming non-foveated regions at the edges of descriptors. *default: 0.1*
- `exclude_self`: Exclude annotations with provenance 'ZEIR', which is the current provenance for ZephIR itself. Effectively, this allows you to do the following: if True, you can track iteratively with separate calls of ZephIR without previous results affecting the next; if False, you can use the previous results as partial annotations for the next. *default: True*
- `exclusive_prov`: Only include annotations with this provenance.
- `fovea_sigma`: Sigma for Gaussian mask over foveated regions of the descriptors. Decreasing this can help prioritize keeping a keypoint at the center. Increase to a large number or set at -1 to disable. *default: 2.5*
- `gamma`: Coefficient for gamma correction. Integrated into `get_data`. *default: 2*
- `grid_shape`: Size of the image descriptors in the xy-plane in pixels. Increasing this may provide a better view of the neighboring features and avoid instabilities due to empty (only 0's) descriptors, but it will also slow down performance. *default: 25*
- `include_all`: Include *all* existing annotations to save file, even those ignored for tracking. In the case that annotations and ZephIR results have matching worldlines in the same frame, annotations will override the results. *default: True*
    > *TIP*: If using *save_mode='o'*, set this argument to 'True' to avoid losing any previous annotations. On the other hand, using 'False' with *save_mode='w'* may allow you to compare the annotations in 'annotations.h5' to the newly saved results in 'coordinates.h5'.
- `lambda_d`: Coefficient for feature detection loss, &lambda;<sub>D</sub>. This regularization is turned on at the last *n_epoch_d* of each optimization loop with everything else turned off. Set to -1 to disable. *default: -1.0*
- `lambda_n`: Coefficient for spring constant for intra-keypoint spatial regularization, &lambda;<sub>N</sub>. Spring constants are calculated by multiplying the covariance of connected pairs by this number and passing the result through a ReLU layer. The resulting loss is also rescaled to this value, meaning it cannot exceed this value. If a covariance value is unavailable, the spring constant is set equal to this number. *default: 1.0*
    > *TIP:* Increase up to 10.0 for non-deforming datasets. Decrease down to 0.01 or turn off for large motion/deformation. Optimal value tends to be between 1.0-4.0. Set to 0 or -1 if regularization is unnecessary (this can speed up performance).
- `lambda_n_mode`: Method to use for calculating $\mathcal{L}$<sub>N</sub>.
    - 'disp': use inter-keypoint displacements
    - 'norm': use inter-keypoint distances (rotation is not penalized)
    - 'ljp': use a Lenard-Jones potential on inter-keypoint distances (collapsing onto the same position is highly penalized)
    - *default: 'disp'*
- `lambda_t`: Coefficient for temporal smoothing loss, &lambda;<sub>T</sub>, enforcing a 0th-order linear fit for intensity over *n_frame* frames. *default: -1.0*
    > *TIP:* 0.1 generally matches order of magnitude of registration loss. Increase up to 1.0 for non-deforming datasets. Set to 0 or -1 if regularization is unnecessary (this will skip the regularization step entirely and can dramatically speed up performance). Alternatively, setting n_frame to 1 will also disable this.
- `load_nn`: Load in spring connections as defined in "nn_idx.txt" if available, save a new one if not. This file can be edited to manually define the connections by worldline ID. The first column is connected to all proceeding columns. *default: True*
    > *WARNING*: Note that all connections are necessarily symmetric (*i.e.*, if object #0 connected to #2, then object #2 must also be connected to #0) even if not defined as such in the file due to how gradients are calculated and accumulated during optimization.
- `lr_ceiling`: Maximum value for initial learning rate. Note that learning rate decays by a factor of 0.5 every 10 epochs. *default: 0.2*
    > *TIP:* If motion is small, set lower to ~0.1. Can use with clip_grad, but may be redundant.
- `lr_coef`: Coefficient for initial learning rate, multiplied by the distance between current frame and its parent. *default: 2.0*
- `lr_floor`: Minimum value for initial learning rate. *default: 0.02*
- `motion_predict`: Enable parent-child flow field to predict low-frequency motion and initialize new keypoints positions for current frame. Requires partial annotations for that frame. *default: False*
    > *TIP:* Identify and annotate a critical subset of keypoints with large errors. These along with motion_predict can dramatically improve tracking quality. Note that this particular flow field does *not* affect descriptors to avoid distortion or image artifacts.
- `n_chunks`: Number of steps to divide the forward pass into. This trades some computation time to reduce maximum memory required. *default: 10*
- `n_epoch`: Number of iterations for image registration, $\mathcal{L}$<sub>R</sub>. *default: 40*
- `n_epoch_d`: Number of iterations for feature detection regularization, $\mathcal{L}$<sub>D</sub>. *default: 10*
- `n_frame`: Number of frames to analyze together for temporal loss, $\mathcal{L}$<sub>T</sub> (see `lambda_t`). Set to 1 if regularization is unnecessary. *default: 1*
- `n_ref`: Manually set the number of keypoints. Leave out to set the number as the maximum number of keypoints available in an annotated frame.
    > *WARNING*: This requires at least one annotated frame with exactly `n_ref` keypoints. The `worldline_id`s from the first frame with exactly `n_ref` keypoints are used to pull and sort annotations from other annotated frames.
- `nn_max`: Maximum number of neighboring keypoints to be connected by springs for calculating &lambda;<sub>N</sub>. *default: 5*
- `save_mode`: Mode for saving results. 
    - 'o' will overwrite existing 'annotations.h5' file.
      > *WARNING*: While provenance can ensure manual annotations remain intact and separable from ZephIR results, this is still quite volatile! Backup of the existing 'annotations.h5' is created before saving. Consider enabling *include_all*. 
    - 'w' will write to a new 'coordinates.h5' file and replace any existing file. 
    - 'a' will append to existing 'coordinates.h5' file. 
    - *default: 'o'*
- `sort_mode`: Method for sorting frames and determining parent-child branches. 
    - 'similarity' minimizes distance between parent and child.
    - 'linear' branches out from reference frames linearly forwards and backwards, with every parent-child one frame apart, until it reaches the first frame, last frame, or another branch. Simplest and fastest.
    - 'depth' uses shortest-path grid search, then sorts frames based on depth in the resulting parent-child tree. This can scale up to *O(n<sup>4</sup>)* in computation with number of frames.
    - *default: 'similarity'*
- `t_ignore`: Ignore these frames during registration. Leave out to analyze all frames.
    > *TIP*: If excluding a large range of frames, you can use `list(range(start, end))` instead of listing every frame. Note that ZephIR uses Python's `eval` function to interpret your input, so any other string expressions compatible with `eval` will work here as well.
- `t_ref`: Only search these frames for available annotations. Leave out if you want to process all annotations.
- `wlid_ref`: Identify specific keypoints to track by *worldline_id* (note: *worldline_id* and *track ID* are used synonymously). Pulls all available annotations for these worldlines. Leave out to track all available keypoints.
    > *WARNING*: This will supercede n_ref.

    > *TIP*: If specifying a large range of worldlines, you can use `list(range(start, end))` instead of listing every worldline. Note that ZephIR uses Python's `eval` function to interpret your input, so any other string expressions compatible with `eval` will work here as well.
- `z_compensator`: Multiply gradients in the z-axis by (1 + `z_compensator`). Since the internal coordinate system is rescaled from -1 to 1 in all directions, gradients in the z-axis may be too small when there is a large disparity between the xy- and z-shapes of the dataset, and thus fail to track motion in the z-axis. Increasing this will compensate for the disparity. Note that gradients will still be clipped to (`clip_grad` * `z_compensator`) if `clip_grad` is enabled. Set to 0 or -1 to disable. *default: -1*

