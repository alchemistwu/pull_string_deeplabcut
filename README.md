# pull_string_deeplabcut
A simple filter for smoothing data.
* 1. Install dependencies

  * Make sure those python packages are already installed:[numpy, pandas, matplotlib, os, argparse]
  * use command: `conda install [package name]`

* 2. Run the script in terminal by command
  * Make sure the directories you entry are in the right format.(Notice that Windows and Linux are different) 

* 3. Linux example:
  *  `python process_deeplabcut.py --i "/home/silasi/mathew/data/Set 1 - Raw.csv" --o "/home/silasi/mathew/output" --s True`
* 4. How it works:
  * 1. Replace the frames with low confidence by zero.
  * 2. Fill the blank frames by the linear approximation of previous frame and the next frame.
  * 3. Apply the lowpass filter.
  * 4. Locate the peaks and drops, get the raw locations.
  * 5. Based on the raw locations, find local max value or local min value close to the raw locations.
