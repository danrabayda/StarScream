import subprocess
import pandas as pd
import numpy as np
import os
from StarScreamLib.functions import vdir, vdirs

#I used pip3 to install youtube-dl and apt to install ffmpeg (I tried using apt for youtube-dl but that version came with a bunch of errors for some reason)
#also make sure to put something in path below. I totally missed it my first run and it took a minute before I realized what was going wrong

# directory to store downloads in
data_folder=vdir("data")
metadata_folder=vdirs(data_folder,"metadata")
downloads_folder=vdirs(data_folder,"downloads")
path = vdirs(downloads_folder,"audio_set_mp4s")

# csv is created by the make_download_list notebook
df = pd.read_csv(os.path.join(metadata_folder,'clean_download.csv'), index_col='segment_id') 
for i in range(len(df)):
    raw_link = df.iloc[i,2] #I was troubleshooting and realized it was just the format of these they were an actual string that read "['http...']" and not a list of one so I just had to chop off the beginning and ends as seen below
    initial_link = raw_link[2:-2] #cuts off the first and last 2 of the string
    my_id = df.index[i]
    print(my_id)
    
    # Get and format start and end times
    # The integer at the end of the id gives the start time in milliseconds
    start_in_seconds = int(my_id.split('_')[-1])
    start_in_seconds = int(start_in_seconds / 1000)
    m, s = divmod(start_in_seconds, 60)
    h, m = divmod(m, 60)
    start = "{:02d}:{:02d}:{:02d}".format(h,m,s)
    
    download_name = '{0}/{1}.mp4'.format(path, my_id)
    if (not os.path.exists(download_name)):
        try:
            output = subprocess.check_output(["youtube-dl", "-g", "{0}".format(initial_link)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            output = output.decode('UTF-8')
            # link to get audio
            link = output.split("\n")[1]
            try:
                # downloads 10 seconds of audio starting at "start"
                output = subprocess.check_output(["ffmpeg", "-ss", start, "-i", "{0}".format(link), "-t", "00:00:10", "-c", "copy", "-strict", "-2", download_name], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            except:
                print("ffmpeg failed")
                continue
        except:
            print('youtube-dl failed')
            continue
    else:
        print('Already downloaded!')
