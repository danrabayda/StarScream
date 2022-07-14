# Project STARSCREAM: An Approach to Classifying Unlabeled Time-Series Audio Data

An encoding convolutional neural network (CNN) is used to cluster segments of time-series audio and classify them either automatically or by human-in-the-machine interactive exploration.

### To Run:

1. Git clone this repo, then run "./install_tensorflow" to install everything you need for a linux os. If there is an error try first running "chmod +x install_tensorflow", for subsequent errors restart terminal and run it again until there are none. If you already have a preferred conda env then just install the pip dependencies with pip install -r requirements.txt
2. Run "conda activate tf" then run "jupyter lab"
3. Open Data_Acquisition*.ipynb and run through it to download all the data and convert it to npys. (If you want to use Audio Set data you have to run audioset_download.py for hours to get that full dataset)
4. Finally run StarScream*.ipynb for the network 


There are several additional functions stored in our pip library AudioStudio. The functions can be seen in https://github.com/danrabayda/AudioStudio/tree/main/AudioStudio/functions.py
