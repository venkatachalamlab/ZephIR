## Guide for using the annotator GUI

Use this manual for verifying ZephIR results and providing any additional supervision.


1. Launch the annotator: `annotator --dataset=.`
2. In the "Annotation window" panel, set provenance from `ANTT` to some identifying set of letters and numbers, such as your initials.
3. Bind macros to hotkeys as shown:


| Key | Macro              | Attributes                            |
|-----|--------------------|---------------------------------------|
| 1   | delete_annotations | _this, _now, *                        |
| 2   | insert_annotations |                                       |
| 3   | jump_to_frame      | (a fully annotated reference frame)   |
| 4   | jump_to_frame      | (a frame of choice)                   |
| 5   | change_provenance  | _this, _now, ZEIR,  (your provenance) |

4. Adjust the colored scrollbars for each channel (rgb order) such that all features of interest are visible.
In the image below, the green channel is too dim for us to see the neurons, so we reduce the range to 0-11.

![GCamP range](https://user-images.githubusercontent.com/39420322/164809374-22b53cd8-bac7-4651-a591-d95658133c56.png)


5. ‘r’ allows us to change views and ‘e’ allows us to go to the previous view. 
Ensure that pressing either of the keys will let you toggle between mip and slice view. 
While annotating, mip view may help you find the rough position of your keypoint, but for 3D datasets, be sure to check the slice view to locate it precisely in all dimensions.

![diff_views](https://user-images.githubusercontent.com/39420322/164809473-fe20f198-4878-489f-89c1-c54d68484ea5.png)

6. ‘a’ toggles between showing all annotations in the current slice and showing all annotations in the entire current frame.

![a](https://user-images.githubusercontent.com/39420322/164809711-8d9f12b9-d0fa-4855-b8a1-1a94b13fe489.png)


7. ‘o’ toggles between showing annotations as filled and unfilled circles.
Using filled circles makes it easy to find, particularly for dense clusters, but it can also make it hard to see the actual data behind it.

![o](https://user-images.githubusercontent.com/39420322/164809828-0ac804e5-97dd-413e-802e-bea9e1576fcd.png)

### With the above setup, follow the flowchart to create new annotations.

![chart](https://user-images.githubusercontent.com/39420322/164809932-1ea4c005-6961-47c5-a4d2-ba4d1d135652.png)

1. **Press 3** to start at the reference frame. 
Select a keypoint of interest.  


2. **Press a** to view only the subset of features present in the same slice, thereby reducing the clutter. 


3. **Press o** to switch to unfilled circle and see more clearly the data behind any annotations. 


4. **Press 4** to move to the chosen frame. 
If an annotation (manual or automatically tracked) exists for the chosen keypoint in the new frame, it will automatically be selected. 
  
  From here, there are three cases:

5. a. If there is no annotation for the keypoint in this frame, place the crosshairs/cursor at the correct coordinates, either by using the scrollbar or clicking with your mouse.
**Press 2** to insert a new annotation for the keypoint.

    b. If the annotation exists but is in an incorrect position for the keypoint, **press 1** to delete the current incorrect annotation. 
Place the crosshairs/cursor at the correct coordinates, either by using the scrollbar or clicking with your mouse. 
**Press 2** to insert a new annotation at the correct coordinates. 

    c. If the annotation is in the correct position, **press 5** to change the provenance of the annotation to your own.
This will make ZephIR treat it as a manual annotation, using it to improve results in subsequent runs and preventing it from being overwritten later.


6. **Press o, a** then **press 3**, to return to the base frame. 

Repeat the above 6 steps to annotate all the keypoints.
At any point, use the "Provenance" panel on the right to selectively view annotations with specific provenances.

**Don't forget to save to disk often!**


  
### In case...

1. In case you deleted annotation for keypoint A and already selected another keypoint B, you can:

   a. Use the "Tracks" panel to reselect the keypoint.

   b. Insert a new annotation for keypoint B and change the track ID in the "Annotation" panel to A afterwards.

   c. Return to the reference frame and reselect the keypoint in that frame.


2. In case you realize you've made a mistake, **don’t panic**!
The annotator makes a backup with every save, which can be found in the /backup folder with a timestamp in the filename.
This is also a reason why you should save to disk regularly. 
