# env : conda activate ds_prepro

## -- Process all videos
from ducksoup import ds_process_parallel

print("Starting analysis")

session_name = "first_data_set" # COMPLETE

ds_process_parallel(sources = "data/"+session_name+"/*/recordings/", target_folder="preproc/"+session_name+"/")

print("Finished analysis")