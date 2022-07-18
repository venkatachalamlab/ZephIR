
### Running annotator

1. Open a command-line interface as administrator (command prompt, anaconda prompt, terminal, etc.) and navigate to dataset folder.

2. Enter the following command:
`annotator --dataset=.`
  > *WARNING*: The annotator uses port 5000 as default. macOS 12.3 and above uses this port for AirPlay. You can either disable this in `Settings>Sharing`, or use a different port by specifying `--port=YOURPORT`.
* Once annotator has successfully launched, you should see the following message: `Starting a server on port 5000`

3. Open a web browser and navigate to: `http://localhost:5000/`

![run_annotator.png](https://user-images.githubusercontent.com/39420322/179257619-3d7b7848-e96b-47e5-874a-eb07ddea4753.png)

### Working with annotator

Before annotating your dataset it's always good to do a hard reload, to do so open your browser's console, in Chrome the keyboard shortcut is ctrl+shift+J. Right click on the reload button and select Empty Cache and Hard Reload. You can close the console now.

1. Use the following hotkeys to control the interface:

| Key | Attribute |
| :---- | :------------------------------------------------- |
| f | forward in time by 1 frame |
| shift + f | forward in time by 10 frames |
| d | backwards in time by 1 frame |
| shift + d | backwards in time by 10 frames |
| v | increase z by 0.05 |
| shift + v | increase z by 0.20 |
| ctrl + v | increase z by 0.01 |
| c | decrease z by 0.05 |
| shift + c | decrease z by 0.20 |
| ctrl + c | decrease z by 0.01 |
| r | next view (currently "slice", "mip", or "volume") |
| e | previous view |
| w | next track |
| q | previous track |
| o | toggle fill circles |
| a | toggle all / nearby annotations |
| 0-9 | run selected macro |


2. Use light blue scrollbars or your mouse to navigate around the volume. "slice" view is ideal for finding the exact coordinates of a keypoint.

![scroll.png](https://user-images.githubusercontent.com/39420322/179257772-cea76073-74fd-49a0-9a0d-46cef9f0da33.png)

3. If the image is not bright enough, use colored scrollbars to adjust the lookup table or use the numeric box to adjust the gamma correction.

![scroll_adjust.png](https://user-images.githubusercontent.com/39420322/179257857-e93ad9ff-424c-440d-b79b-cc17f36c5c6b.png)

4. Use hotkeys `d` or `f` to explore different time points. This can also be done using the provided macro: `jump_to_frame`

5. Use hotkeys `q` or `w` to cycle through existing annotations in the current time point.


### Annotating

1. You can add a new annotation by using the provided macro: `insert_annotation`

2. When an annotation is selected, its information are shown in the "Annotation" panel in the top right corner.

![annotation.png](https://user-images.githubusercontent.com/39420322/179257965-be11a446-24ed-4e96-b126-be21ba5b3a9b.png)

3. Each keypoint should have a unique track ID through out the entire dataset. If you want to annotate a different keypoint, create a new track by using the provided macro: `create_track`.


**See [here](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/Guide-annotatorGUI.md) for a more detailed guide on creating new annotations.**


### Macro

Macros are Python functions that can directly interact with the annotations and the GUI state.
In the "Annotation window" panel, you can choose different macros and assign them to hotkeys.
Hover over `args` to see the documentation for the macro and the passed arguments.

Certain GUI state properties can be passed as arguments:

`_now`: current time point shown in the image window.

`_this`: current selected track ID.

`_ACTIVE`: current active provenance (as shown under "prov" at the top of "Annotation window" panel).

`*`: all applicable items.


![macros.png](https://user-images.githubusercontent.com/39420322/179257954-5ab81d82-3457-4d8f-8522-5229cd38b33e.png)

#### Useful provided macros

`insert_annotation`: Adds a new annotation at the current cursor location with the current track ID.

`insert_local_max`: Adds a new annotation at the local maximum nearest to the current cursor location with the current track ID. 
Its arguments specify the search area. 
This is particularly useful for placing annotations at the center of fluorescent cells.
You can have different keys for this macro with different arguments to use them for cells of different size.

* Note: When you click on different parts of the image, you can see their coordinates at the top of the "Annotation window" panel. 
Choose an average-sized cell and take advantage of this feature to get an idea about the dimensions of a cell. 
Use this information to set parameters for this macro.

`jump_to_frame`: Jumps to the specified time point. 
When annotating multiple time points, it is useful to have a key for multiple time points to quickly toggle between them.

`create_track`: Creates a new track and selects it.

#### Useful macros for interfacing with ZephIR

`change_provenance`( *, _now, ZEIR, ANTT ): Updates all neurons in the active frame with provenance "ZEIR" to "ANTT." This effectively promotes the current frame with ZephIR results to a full verified annotation.

`update_frame`( _now, True, False, False ): Given some partial annotations in the current frame, runs ZephIR on the keypoints connected to those annotations in the spring network ONLY. 
This is a fast, efficient way to see how a set of partial annotation can affect its nearest neighbors.
Setting the second argument, restrict_update, to False will run ZephIR on all unannotated keypoints in the frame.

`overwrite_zeir_checkpoint`( nn_max, 10 ): Updates a given keyword argument for ZephIR ("nn_max") with the given value ("10") in the checkpoint.pt file. 
Any updates to arguments for model building/compiling will trigger a recompile and may take longer to complete. 
Use this to update ZephIR parameters when using update_frame.
> *TIP*: Using key/value pair of `(args, all)` will load in all arguments as listed in `args.json`. Use this to edit multiple parameters before running ZephIR again. 

### Saving and Loading

On the top right of the "Annotation window" panel, there are two buttons to save and load annotations from disk. 
Annotations are save to/loaded from the dataset directory.
Saving will overwrite any existing `annotations.h5` file.
