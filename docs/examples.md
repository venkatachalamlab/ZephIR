We tested a number of datasets across various systems. For each dataset, we show the original movie, a movie annotated with the tracking result, a list of parameters used, and a brief explanation for each parameter. Note that only parameters that were changed from the default (as listed [here](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/Guide-parameters.md)) are listed here.


## Simulation of Gaussian cells with uncorrelated random motion.

`lambda_n=-1`: ![uncorr_lambda_n_off](https://user-images.githubusercontent.com/39420322/174839297-2264b999-c8b1-4c9c-bbc7-699ce92ddade.gif)
`lambda_n=1.0`: ![uncorr_lambda_n_on](https://user-images.githubusercontent.com/39420322/174839378-3eb122a0-a53d-4c24-bab1-89f61b04e608.gif)


## Simulation of Gaussian cells with strongly correlated motion.

`lambda_n=-1`: ![corr_lambda_n_off](https://user-images.githubusercontent.com/39420322/174838603-8a0802ba-2502-49ea-a737-3fbb1efae2e7.gif)
`lambda_n=1.0`: ![corr_lambda_n_on](https://user-images.githubusercontent.com/39420322/174839560-7701c938-9431-46ad-ba22-e5034776b464.gif)


## Freely moving *C. elegans*

![annotated](https://user-images.githubusercontent.com/39420322/174844925-698124e6-9dee-4350-9b8d-e3d9d15e76b6.gif)


- `channel=1`: This dataset has 2 channels, but only the second channel has the neurons that we want to track.
- `clip_grad=-1`: This dataset has significant motion in all directions. To accomodate neurons with large displacements between parent and child frames, we disable gradient clipping, relying on learning rates to adjust how much displacement we allow.
- `fovea_sigma=10`: Along with `grid_shape`, this opens up the "view" range for descriptors and de-emphasizes the importance of the center of the descriptor relative to its neighbors. While this can be detrimental for rapidly deforming densely-packed clusters, it is particularly helpful for neurons towards the edges of the volume. 
- `grid_shape=49`: This increases the size of the descriptors. Generally, it should be about ~150% of the cell's size in pixels, but we use much larger descriptors here to avoid having any descriptors with all zeros, which can cause instabilities during gradient descent. This is usually not an issue, but the head swings generate large motions in particularly sparse areas of the volume.
- `lambda_n_mode=norm`: This dataset sees significant rotations in relative positions of neighboring neurons. `norm` mode avoids penalizing those neighbors.
- `lr_ceiling=0.1`: This limits frame-to-frame displacement of each neuron. Since we uncapped the gradient values, we can be a little more aggressive with this parameter.
- `lr_floor=0.01`: We lower this to ensure that parent-child frames that are very close together also produce similar neuron positions. We have good, distinct clusters of similar frames around each reference frame, so we can lower this further.
- `motion_predict=True`: We verify and fix 10 neurons to use as partial annotations across all frames in this dataset. To make full use of these partial annotations, we turn on `motion_predict` to improve tracking for the rest of the neurons.
- `z_compensator=4.0`: This dataset has significant motion in the z-axis even when the center of the volume is fixed, but its size in z is ~1/10 of the xy shape. We increase this parameter to compensate for the disparity.


## Semi-immobilized unc-13 *C. elegans*

![annotated](https://user-images.githubusercontent.com/39420322/174843026-70a87cbd-5794-4d85-9ac7-b12807829346.gif)


- `clip_grad=0.2`: Despite the large jumps for some neurons during pumping events, the dataset as a whole does not exhibit much motion. Clipping the gradients prevents tracking results from becoming wildly inaccurate for the low-motion neurons.
- `lambda_n=0.1`: Most of the neurons here do not show significant motion, but those that do are often isolated and move independently. Lowering this parameter prevents more stationary neurons from being pulled out of position due to nearby high-motion neurons.
- `lr_ceiling=0.1`: Along with `clip_grad`, this prevents tracking results from moving too much frame-to-frame.
- `lr_floor=0.01`: Some parent-child pairs do not see any motion at all. We lower this parameter to ensure the neuron positions also do not move for those frames.


## Immobilized NeuroPAL *C. elegans*

![annotated](https://user-images.githubusercontent.com/39420322/174843090-1235e381-73ac-4841-94ad-48b9f2a22caa.gif)


- `gamma=2.5`: Slight increase in gamma helps filter out noise.
- `lambda_n=2.0`: This dataset only exhibits small, but well-correlated motion (all neurons basically move the same way). We increase the spring constant to reflect this, especially since we only have one reference frame for this dataset and thus do not have covariance values to tune individual spring connections.
- `lr_ceiling=0.1`: This dataset has very little motion overall. Along with other learning rate parameters, it is lowered to reflect this.
- `lr_coef=1.0`: This dataset has very little motion overall. Along with other learning rate parameters, it is lowered to reflect this.
- `lr_floor=0.01`:  This dataset has very little motion overall. Along with other learning rate parameters, it is lowered to reflect this.
- `sort_mode=linear`: This dataset only has one reference frame (t=0) and only samples a single posture. Sorting by similarity is unnecessary, so we simply go in chronological order. It is worth noting that linear sort mode is fastest to compute and easiest to interpret.
- `z_compensator=0.5`: This dataset has a slight displacement in z, but its size in z is ~1/10 of the xy shape. We increase this parameter slightly to ensure we can capture the z-displacement.


## Chinese Hamster Ovarian nuclei

![annotated](https://user-images.githubusercontent.com/39420322/174843166-d858bf4c-72dd-4f73-a3cc-b875c51f5468.gif)


> Data available at: http://celltrackingchallenge.net/3d-datasets/

- `clip_grad=0.33`: This dataset exhibits large fluctuations in parent-child frame similarities. We increase the learning rates parameters to accomodate the larger range, but we reduce the gradient values here to prevent tracking results from moving too much.
- `fovea_sigma=40`: We track very large cells for this dataset. Along with `grid_shape`, this parameter is increased to capture the entire cell in the descriptor.
- `grid_shape=125`: We track very large cells for this dataset. Along with `fovea_sigma`, this parameter is increased to capture the entire cell in the descriptor.
- `lambda_n=-1`: Cells in this dataset undergo mitosis. Zephir is built to track a fixed number of keypoints, but we can accomodate the mitosis events by starting from the last frame with the maximum number of cells and allowing keypoints to collapse together as we move backwards. We disable spring connections to avoid penalizing collapsing keypoints.
- `lr_ceiling=0.4`: This dataset exhibits large fluctuations in parent-child frame similarities. Along with `lr_floor`, we increase this to accomodate the larger range.
- `lr_floor=0.06`: This dataset exhibits large fluctuations in parent-child frame similarities. Along with `lr_ceiling`, we increase this to accomodate the larger range.
- `sort_mode=linear`: Cells in this dataset undergo mitosis. Zephir is built to track a fixed number of keypoints, but we can accomodate the mitosis events by starting from the last frame with the maximum number of cells and allowing keypoints to collapse together as we move backwards. This means temporal ordering becomes important, so we set this parameter to linear.


## Hydra

![annotated](https://user-images.githubusercontent.com/39420322/174845524-f032ea0f-b731-49ee-9cdb-d67e59192a19.gif)


> Data available at: https://www.ebi.ac.uk/biostudies/studies/S-BSST428

- `allow_rotation=True`: Features in this dataset show clear rotation, especially in the tentacles. To accomodate this, we enable optimization for an additional parameter that controls rotation of descriptors in the xy-plane.
- `grid_shape=11`: Each neuron in this dataset is small, so a lower value here is sufficient.
- `lambda_n=0.1`: While the tentacles create a good biological scaffolding for the neurons, they deform and stretch over time. We lower the spring constants to accomodate these deformations.
- `lambda_n_mode=norm`: This dataset shows clear rotations in relative positions of neighboring neurons, especially in the tentacles. `norm` mode avoids penalizing those neighbors.
- `lr_floor=0.08`: The large, bright body obscures the changes in the tentacles when calculating parent-child frame similarities. We increase this parameter to compensate.
- `sort_mode=linear`: This dataset does not repeatedly sample the same postures, but rather continuously deforms over time. To reflect this, we track linearly.


## Behaving mouse

![annotated](https://user-images.githubusercontent.com/39420322/174844169-fcb7c470-e2a8-4317-aa7e-cb298f5559b8.gif)


> Data available at: https://ibl.flatironinstitute.org/public/churchlandlab/Subjects/CSHL047/2020-01-20/001/raw\_video\_data/

- `allow_rotation=True`: Features in this dataset show clear rotation, especially in the paws. To accomodate this, we enable optimization for an additional parameter that controls rotation of descriptors in the xy-plane.
- `dimmer_ratio=0.8`: Unlike fluorescent microscopy data, this is a particularly feature-rich dataset. Increasing this parameter emphasizes the neighboring features relative to the centers of descriptors.
- `fovea_sigma=49`: We track very large features for this dataset. Along with `grid_shape`, this parameter is increased to capture the entire body part in the descriptor.
- `grid_shape=65`: We track very large features for this dataset. Along with `fovea_sigma`, this parameter is increased to capture the entire body part in the descriptor.
- `lambda_n=0.1`: While the skeleton creates a good biological scaffolding for the mouse, this dataset lacks a third dimension and thus the distances between body parts are not well-preserved in the image. We lower the spring constants to accomodate this.
- `lambda_n_mode=norm`: This dataset shows clear rotations in relative positions of neighboring features, especially in the paws. `norm` mode avoids penalizing those neighbors.
- `nn_max=3`: We only track 10 keypoints for this dataset. We reduce the number of maximum neighbors to reflect the small number of total points.

