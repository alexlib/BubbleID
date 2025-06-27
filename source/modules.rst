BubbleID Module
===============
.. role:: red

**class** :red:`DataAnalysis`

This class is for generating and analyzing data from a single pool boiling experiment or specific video from an experiment.

    :red:`__init__(...)`
        Parameters:
            * imagesfolder: path to folder where .jpg images are saved
            * videopath: path to where .avi video is saved
            * savefolder: path to folder where all data will be saved
            * extension: unique identifier that will be attached to each file saved in the save folder
            * modelweightsloc: path to location where model weights are saved
            * device: ('cpu' or 'gpu')

    :red:`GenerateData(...)`
        Parameter:
            * thres: Value between 0 and 1 that specifies the cut off of confidence of detected bubbles that are kept
        Output:
            * bb-Boiling-output-{extension}.txt: File containing the coordinates of the bounding boxes of the detected bubbles and the confidence.
            * vapor_{extension}.npy: Numpy file containing the count of pixels containing a bubble. This file can be used for approximating vapor faction.
            * bubble_size_bt-{extension}.npy: Numpy file containing the number of pixels each detected bubble consists of for each frame. This can be used to determine bubble count and size.
            * bubind_{extension}.npy: Numpy file containing the labeled index of each bubble
            * frames_{extension}.npy: Numpy file containing the frames each bubble is present in
    :red:`Plotvf()`
        Output:
            * vaporfig_{extension}.npy: Plot of vapor fraction over the duration of frames along with the rolling sample average
    :red:`Plotbc()`
        Output:
            * bcfig_{extension}.npy: Plot of the bubble count over the duration of frames and rolling sample average
    :red:`PlotInterfaceVelocity(...)`
        Parameter:
            * bubble: index of the bubble to find the interface velocity.
        Output:
            * displays an image of the frame and bubble contour to check if this is the intended bubble
            * velocity_{extension}_{bubble}.png: Figure of the interface velocity along the bubble contour vs frames bubble is present in. This figure has been smoothed with a gaussian filter and has a range of -30 to 30. This function is best used with higher framerates (i.e. 3000 fps) and larger bubbles.




