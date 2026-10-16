from video_processing import combine_videos, get_movie_duration, change_frame_rate, extract_audio, combine_audio, replace_audio, extract_sub_video_sentences, re_encode, combine_2_videos
from conversions import get_file_without_path
try:
    from .align import align_folder
except ImportError:
    from align import align_folder
from pathlib import Path
import glob
import os
import numpy as np
import datetime

#Macros to parse DuckSoup name, change this to correspondent position if needed
date_position_in_file_name = 3
hour_position_in_file_name = 4
player_position_in_file_name = -4
dyad_position_in_file_name = -6


#Parse FILE NAMES
def get_player(file):
    file_tag = get_file_without_path(file)
    return file_tag.split("-")[player_position_in_file_name]

def get_dyad(file):
    file_tag = get_file_without_path(file)
    return file_tag.split("-")[dyad_position_in_file_name]

def get_date(file):
    file_tag = get_file_without_path(file)
    return file_tag.split("-")[date_position_in_file_name]

def get_hour(file_tag):
    return file_tag.split("-")[hour_position_in_file_name]


##-- trim_folder
def trim_folder(source_folder, folder_tag, target_folder, trimed_path, extension=".mp4", verbose=True):
    """
    | LEGACY, kept for single_pass=False. Use align.align_folder instead.
    | This function takes the start of each recording from its file name (the time the participant joined,
    | not the time the recording started) and cuts every file to start at the latest file start of the
    | folder. This shifts the partners relative to each other by the difference of their join times, and
    | when a participant reconnects (-c-2-, ...) everything recorded before the reconnection is lost.
    """
    #Compute the delays in the videos
    if verbose:
        print("Computing video deltas with ffmpeg...")

    d = {}
    for file in glob.glob(source_folder + "*" + extension):
        if verbose:
            print(file)
            print()

        file_tag = get_file_without_path(file, with_extension=True)
        date = file_tag.split("-")[date_position_in_file_name]
        hour = file_tag.split("-")[hour_position_in_file_name]
        val = datetime.datetime.strptime(date + " " + hour, "%Y%m%d %H%M%S.%f")
        d[file_tag] = val

    if verbose:
        print(d)
        print()

    max_time = max(d.values())
    for k in d.keys():
        val = d[k] - max_time
        val = abs(val.total_seconds())
        d[k] = val

    #trim videos and save
    if verbose:
        print("Triming videos with ffmpeg...")

    if not os.path.isdir(target_folder + trimed_path):
        try:
            os.mkdir(target_folder + trimed_path )
        except:
            pass

    for file in glob.glob(source_folder + "*" + extension):
        if verbose:
            print("Triming file : "+ file)

        file_name = get_file_without_path(file, with_extension=True)
        start = d[file_name]
        length = get_movie_duration(file) - start
        if length <= 0:
            continue

        start = str(datetime.timedelta(seconds=start))
        if not os.path.isdir(target_folder + trimed_path + folder_tag):
            os.mkdir(target_folder + trimed_path + folder_tag)

        extract_sub_video_sentences(source_name=file, target_name=target_folder + trimed_path + folder_tag + file_name, start=str(start), length=str(length))


## --re_encode_folder 
def re_encode_folder(source_folder, folder_tag, target_folder, re_encode_path, extension=".mp4", resolution="1280:720", preset="veryslow", crf = 18, verbose=True):
    if verbose:
        print("Re-encoding videos with ffmpeg...")
        print()

    if not os.path.isdir(target_folder + re_encode_path):
        try:
            os.mkdir(target_folder + re_encode_path)
        except:
            pass            

    for file in glob.glob(source_folder +"*" + extension):
        if verbose:
            print("re-encoding : "+ file)

        if not os.path.isdir(target_folder + re_encode_path + folder_tag):
            os.mkdir(target_folder+ re_encode_path + folder_tag)
            
        file_tag = get_file_without_path(file)
        output = target_folder+ re_encode_path + folder_tag + file_tag + ".mp4"
        re_encode(source=file, output=output, resolution=resolution, preset=preset, crf = crf)


## --re_encode_folder 
def change_fps_folder(source_folder, folder_tag, target_folder, change_fps_path="new_fps/", extension=".mp4", target_fps=30, verbose=True):
    if verbose:
        print("Changing FPS of videos with ffmpeg...")
        print()

    if not os.path.isdir(target_folder + change_fps_path):        
        os.makedirs(target_folder + change_fps_path, exist_ok=True)

    for file in glob.glob(source_folder +"*" + extension):
        if verbose:
            print("re-encoding : "+ file)

        if not os.path.isdir(target_folder + change_fps_path + folder_tag):
            os.mkdir(target_folder+ change_fps_path + folder_tag)
            
        file_tag = get_file_without_path(file)
        output = target_folder+ change_fps_path + folder_tag + file_tag + ".mp4"

        change_frame_rate(source=file, output=output, target_fps=target_fps, overwrite=True)


##-- stack_videos
def stack_videos(files, output, tile="640:480"):
    """
    | Check video: 2 videos side by side, or 4 in a 2x2 grid (order: top left, top right, bottom left, bottom right).
    | Each input is scaled to fit the tile (aspect ratio kept, padded), so inputs of different sizes can be stacked.
    | Video only; the audio is added by combine_folder.
    """
    import subprocess
    w, h = tile.split(":")
    fit = "".join("[%d:v]scale=%s:%s:force_original_aspect_ratio=decrease,pad=%s:%s:(ow-iw)/2:(oh-ih)/2,setsar=1[s%d];"
                  % (i, w, h, w, h, i) for i in range(len(files)))
    if len(files) == 4:
        stack = "[s0][s1]hstack[t];[s2][s3]hstack[b];[t][b]vstack[v]"
    elif len(files) == 2:
        stack = "[s0][s1]hstack[v]"
    else:
        raise ValueError("stack_videos needs 2 or 4 videos")
    cmd = ["ffmpeg", "-y"]
    for f in files:
        cmd += ["-i", f]
    cmd += ["-filter_complex", fit + stack, "-map", "[v]", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-shortest", output]
    subprocess.call(cmd)


##-- combine_folder
def combine_folder(source_folder, folder_tag, target_folder, combined_path, combined_with_audio_path,extension= ".mp4", combine_audio_flag=True, verbose=True):
    if verbose:
        print("combining videos with ffmpeg...")
        print()
    
    combined_video_name = folder_tag.split("/")[-2]
        
    #Combine videos
    nb_files = len(glob.glob(source_folder + "*" + extension))
    if nb_files != 4 and nb_files != 2 :
        print("I can only combine 4 or 2 videos, nb file is "+ str(nb_files))
        print()
        return
    
    #Get organised files in the order (dry, wet, dry wet)
    files = glob.glob(source_folder + "*"+ extension)
    players = []
    for file in files:
        players.append(get_player(file))
    players = np.unique(players)
    dyad = get_dyad(file)
    manipulations = ("dry", "wet")
    organised_files = []
    for player in players:
        for manipulation in manipulations:
            file_list = glob.glob(source_folder+"*"+dyad+"*"+"-u-"+player+"*"+manipulation+ extension)
            if len(file_list)>0:
                organised_files.append(file_list[0])
    
    if not os.path.isdir(target_folder + combined_path):
        os.mkdir(target_folder+ combined_path)
    
    output = target_folder+combined_path + combined_video_name + extension
    #Dyads with manipulated and non-manipulated
    #(inputs are scaled to a common tile, as the single-pass outputs keep each participant's native resolution)
    if len(organised_files) in (2, 4):
        stack_videos(organised_files, output)


    #extract and combine audios
    if combine_audio_flag:
        import uuid 
        if not os.path.isdir(target_folder + combined_with_audio_path):
            os.mkdir(target_folder + combined_with_audio_path)

        audios = []
        files = glob.glob(source_folder + "*"+ extension)
        for cpt1, file in enumerate(files):
            audio_name = str(cpt1) + str(uuid.uuid1()) +"____.wav"
            extract_audio(file, audio_name)
            audios.append(audio_name)

        aux_wav = str(uuid.uuid1()) + ".wav"
        combine_audio(audios, aux_wav)

        #replace audio
        file_tag = get_file_without_path(output)
        replace_audio(output,  aux_wav, target_folder + combined_with_audio_path + file_tag + extension)

        for audio in audios:
            os.remove(audio)
        
        os.remove( aux_wav)


def ds_process_audio_only(source_folder
                            , folder_tag
                            , target_folder="audio_preproc/"
                            , extension=".ogg"
                            , trimed_path="trimed/"
                            , combined_path = "combined_audio/"
                            , single_pass=True
                            , log_folder=None
                            , duration=None
                            , verbose=True):
    """
    | Aligns the audio recordings of one interaction and mixes the two participants (dry and wet).
    | single_pass=True (default): align.align_folder, aligned on the recording starts read from the DuckSoup
    |     logs, reconnections joined, written as lossless WAV files in trimed_path/folder_tag.
    | single_pass=False: legacy trim_folder (see its docstring for the alignment problems).
    """
    from transform_audio import combine_audio_files

    os.makedirs(target_folder, exist_ok=True)
    nb_files = len(glob.glob(source_folder + "*" + extension))
    if nb_files == 0 :
        return

    # trim audio files so they are in sync
    if single_pass:
        align_folder(source_folder=source_folder, folder_tag=folder_tag, target_folder=target_folder
                     , trimed_path=trimed_path, extension=extension, log_folder=log_folder, duration=duration
                     , video_out=False, write_wav=True, verbose=verbose)
        extension = ".wav"
    else:
        trim_folder(source_folder=source_folder
                    , folder_tag=folder_tag
                    , target_folder=target_folder
                    , trimed_path=trimed_path
                    , extension=extension
                    , verbose=verbose
                    )

    # 
    source_folder = target_folder + trimed_path + folder_tag
    
    if verbose:
        print("combining audios...")
        print()
        
    #Combine audios
    nb_files = len(glob.glob(source_folder + "*" + extension))
    if nb_files != 4 and nb_files != 2 :
        print("I can only combine 4 or 2 videos, nb file is "+ str(nb_files))
        print()
        return
    

    os.makedirs(target_folder+ combined_path+folder_tag, exist_ok=True)    
    
    #Get (dry files and wet files)
    files = glob.glob(source_folder + "*"+ extension)
    players = []
    for file in files:
        players.append(get_player(file))
    players = np.unique(players)

    #For dry and wet:
    for manipulation in ("dry", "wet"):
        #Get all dry or wet files
        file_list = glob.glob(source_folder+"*"+manipulation + extension)
        if len(file_list) == 2:
            #combine files in organised_files
            print(file_list)
            combine_audio_files(file_list, target_folder + combined_path + folder_tag + manipulation + extension)

def ds_process(source_folder
                            , folder_tag
                            , target_folder="preproc/"
                            , extension=".mp4"
                            , trimed_path="trimed/"
                            , re_encode_path = "re-encode/"
                            , combine_videos=True
                            , combined_path="combined/"
                            , combined_with_audio_path="with_audio/"
                            , change_fps_path = "new_fps/"
                            , combine_audio_flag=True
                            , verbose=True
                            , resolution="1280:720"
                            , preset="veryslow"
                            , crf = "18"
                            , target_fps = "30"
                            , single_pass=True
                            , crf_single_pass=0
                            , log_folder=None
                            , duration=None
                            , write_wav=True
                            , ffmpeg_threads=0
                            ):
    """
    | A function to preprocess recordings made with ducksoup.
    |
    |
    | This function trims, re-encodes, changes the fps and combines the videos collected by ducksoup
    | input:
    |   source_folder : folder where the videos to pre process are located (with a slash at the end)
    |   folder_tag    : is the identifier of the interaction, where all processed videos will be saved
    |   target_folder : is the name of the folder where all the processing will be saved. With a slash in the end. Usually prepoc/
    |   extension : the extension of the videos to process
    |   trimed path : the name of the folder that will be created inside target folder
    |   combine_videos : whether the function should combine videos after preprocessing together into one single video with all four videos. 
    |        This is useful to make sure that videos are in sync.
    |    combine_path : the name of the folder to use when combining the files
    |    verbose : whether to print a detailed output of the stages of the preprocessing
    |    resolution="1280:720"
    |    preset="veryslow" This is the preset used by open CV
    |    crf = "18" CRF is the factor quality. CRF : use 18 for almost lossless. Lower is better but takes more size. More info here : https://superuser.com/questions/1556953/why-does-preset-veryfast-in-ffmpeg-generate-the-most-compressed-file-compared
    |    target_fps = "30" Target FPS to resample videos
    |    single_pass=True : align and encode in ONE ffmpeg pass with align.align_folder (default):
    |        aligned on the recording starts read from the DuckSoup logs (log_folder, default: the parent of
    |        source_folder), reconnection segments joined on one timeline with the gaps filled with black frames
    |        and silence, native resolution (resolution and re_encode_path are not used), constant target_fps,
    |        x264 at crf_single_pass (0 = lossless), a lossless WAV next to each mp4 (write_wav), and a manifest
    |        trimed_path/folder_tag/alignment.csv with the gaps and the covered time of each output.
    |        duration: optional maximum length of the outputs (s). ffmpeg_threads: threads per ffmpeg (0 = auto).
    |    single_pass=False : legacy chain re-encode -> change fps -> trim_folder (three lossy encodes, and the
    |        alignment problems described in trim_folder).
    | Usage Example: 
    |
    | from ducksoup_preproc import ds_process
    | import glob
    | for source_folder in glob.glob("videos/*/recordings/"):
    |    folder_tag = source_folder.split("/")[-3] + "/"
    |    ds_process(source_folder=source_folder, folder_tag=folder_tag)
    |   
    | Also, this function can be used in parallel with ds_process_parallel
    """
    # create target folder if it doesn't exist, if it exists, pass
    if not os.path.isdir(target_folder):
        try:
            import errno
            os.mkdir(target_folder)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass
    
    if verbose:
        print(folder_tag)
        print()
    
    if single_pass:
        align_folder(source_folder=source_folder, folder_tag=folder_tag, target_folder=target_folder
                     , trimed_path=trimed_path, extension=extension, log_folder=log_folder, duration=duration
                     , write_wav=write_wav, verbose=verbose, fps=target_fps, crf=crf_single_pass
                     , threads=ffmpeg_threads)
        if combine_videos:
            combine_folder(source_folder=target_folder + trimed_path + folder_tag
                           , folder_tag=folder_tag
                           , target_folder=target_folder
                           , combined_path=combined_path
                           , combined_with_audio_path=combined_with_audio_path
                           , extension=".mp4"
                           , verbose=verbose
                           , combine_audio_flag=combine_audio_flag
                           )
        return

    #reencode
    re_encode_folder(source_folder=source_folder
                     , folder_tag=folder_tag
                     , target_folder=target_folder
                     , re_encode_path=re_encode_path
                     , extension=extension
                     , resolution=resolution, preset=preset
                     , crf = crf
                     , verbose=verbose
                     )
    
    #change fps
    source_folder = target_folder + re_encode_path + folder_tag
    change_fps_folder(source_folder=source_folder
                      , folder_tag=folder_tag
                      , target_folder=target_folder
                      , change_fps_path=change_fps_path
                      , extension=extension
                      , target_fps=target_fps
                      , verbose= verbose
                      )

    #trim videos
    source_folder = target_folder + change_fps_path + folder_tag
    trim_folder(source_folder=source_folder, folder_tag=folder_tag, target_folder=target_folder, trimed_path=trimed_path, extension=extension, verbose=verbose)

    #Combine videos
    if combine_videos:
        source_folder = target_folder + trimed_path + folder_tag
        combine_folder(source_folder=source_folder
                       , folder_tag=folder_tag
                       , target_folder=target_folder
                       , combined_path=combined_path
                       , combined_with_audio_path=combined_with_audio_path
                       , extension=extension
                       , verbose=verbose
                       , combine_audio_flag=combine_audio_flag
                       )


## Parallel processing functions
def parallelize_function(source_folder, folder_tag_idx =-3
                                    , target_folder="preproc/" 
                                    , extension=".mp4"
                                    , trimed_path="trimed/"
                                    , re_encode_path = "re-encode/"
                                    , combine_videos=True
                                    , combined_path="combined/"
                                    , combined_with_audio_path="with_audio/"
                                    , combine_audio_flag=True
                                    , verbose=True
                                    , resolution="1280:720"
                                    , preset="slow"
                                    , crf = "18"
                                    , target_fps=30
                                    , single_pass=True
                                    , ffmpeg_threads=0
                                    ):
    
    folder_tag = Path(source_folder).parts[folder_tag_idx] + "/"
    print("the folder tag is : " + folder_tag)

    ds_process(source_folder = source_folder
                    , folder_tag = folder_tag
                    , target_folder=target_folder
                    , extension=extension
                    , trimed_path=trimed_path
                    , re_encode_path = re_encode_path
                    , combine_videos=combine_videos
                    , combined_path=combined_path
                    , combined_with_audio_path=combined_with_audio_path
                    , combine_audio_flag=combine_audio_flag
                    , verbose=verbose
                    , resolution=resolution
                    , preset=preset
                    , crf = crf
                    , target_fps=target_fps
                    , single_pass=single_pass
                    , ffmpeg_threads=ffmpeg_threads
                    )

def ds_process_parallel(sources , folder_tag_idx =-3
                                    , target_folder="preproc/" 
                                    , extension=".mp4"
                                    , trimed_path="trimed/"
                                    , re_encode_path = "re-encode/"
                                    , combine_videos=True
                                    , combined_path="combined/"
                                    , combined_with_audio_path="with_audio/"
                                    , combine_audio_flag=True
                                    , verbose=True
                                    , resolution="1280:720"
                                    , preset="slow"
                                    , crf = "18"   
                                    , target_fps=30                     
                                    , single_pass=True
                                    , processes=None
                                    , ffmpeg_threads=0
                                    ):
    """
        from ducksoup import ds_process_parallel
        folder_tag_idx : is the place where the * is, in the path. The place where recoridngs are blocked by.
        ds_process_parallel(sources = "videos/latest_30_10_b/test_2/*/recordings/")
        check help(ds_process) for other arguments
        processes : number of interactions processed in parallel (default: all cores of the machine; on a cluster,
            use the cores allocated to the job, e.g. int(os.environ["SLURM_CPUS_PER_TASK"]) // ffmpeg_threads)
        ffmpeg_threads : threads per ffmpeg process (0 = auto, i.e. all cores)

    """
    import multiprocessing
    from itertools import repeat
    
    pool_obj = multiprocessing.Pool(processes)
    a_args     = glob.glob(sources)
    
    pool_obj.starmap(parallelize_function, zip(a_args
                                  , repeat(folder_tag_idx)
                                  , repeat(target_folder)
                                  , repeat(extension)
                                  , repeat(trimed_path)
                                  , repeat(re_encode_path)
                                  , repeat(combine_videos)
                                  , repeat(combined_path)
                                  , repeat(combined_with_audio_path)
                                  , repeat(combine_audio_flag)
                                  , repeat(verbose)
                                  , repeat(resolution)
                                  , repeat(preset)
                                  , repeat(crf)
                                  , repeat(target_fps)
                                  , repeat(single_pass)
                                  , repeat(ffmpeg_threads)
                                  )
                    )