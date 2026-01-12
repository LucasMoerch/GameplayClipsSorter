# VideoSorter (CS / Rocket League / Other)
Sort gameplay clips in the same folder as sorter.py into *sorted/cs*, *sorted/rocket_league*, and s*orted/other games* using OpenCV template matching (cv2.matchTemplate).

## Setup
Install dependencies:

    pip install opencv-python numpy

Place files like this:

    your_video_folder/
	    sorter.py
	    templates/

**Run**

    python sorter.py

### Notes
Templates should be tight crops of stable HUD elements.

Tune SAMPLES_PER_VIDEO, THRESHOLDS, MIN_HITS_BY_LABEL, STRONG_SCORE if needed.

The script moves files into sorted/ (not copies).
