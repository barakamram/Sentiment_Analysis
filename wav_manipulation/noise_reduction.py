import os
import wave
from random import random, randint

import librosa
import soundfile

wav_dir = ""
path_noises = ""


def get_duration(wav_file):
    """
    input:
        wav_file: (string) path to file
    output:
        duration: (int) duration of the wav file
    """
    with wave.open(wav_file, 'rb') as wf:
        frame_rate = wf.getframerate()
        num_frames = wf.getnframes()

        duration = num_frames / frame_rate  # Calculate the duration
    return duration


def count_dir(path_dir):
    """
    input:
        path_dir: (string) path to directory
    output:
        count: (int) how many files are in this directory
    """
    count = 0
    # Iterate directory
    for path in os.listdir(path_dir):
        # check if current path is a file
        if os.path.isfile(os.path.join(path_dir, path)):
            count += 1

    return count


def get_next_noise(noises_path, current_idx):
    """
    input:
        current_noise: (string) path to noises
        current_index:  (int) represent the current index in the folder
    output:
        cur_noise: (string) represent the next noise in the folder
        cur_index: (int) return the new current_index
    """

    cur_index = randint(0, count_dir(noises_path) - 1)
    if cur_index == current_idx:
        get_next_noise(noises_path, current_idx)

    cur_noise = os.listdir(noises_path)[cur_index]
    return cur_noise, cur_index


def resize_noise(wav_path, duration_seconds):
    """
    input:
        wav_path: (string) to be cut
        duration_seconds: (seconds) what length will be the temporary noise
    return:
          path: (string) path of the temp noise audio after adjust the time
    """

    audio, sr = librosa.load(wav_path, sr=44100)
    splited_filename = audio[:int(duration_seconds * sr)]  # Truncate the audio signal to the desired duration

    wav_name_with_type = wav_path.split("\\")[-1]
    wav_name_without_type = wav_name_with_type.split(".")[0]

    path_temp_noises = f'.\\..\\data\\training\\temp_noises'
    if not os.path.exists(path_temp_noises):
        os.mkdir(path_temp_noises)

    path = os.path.join(path_temp_noises, f'{wav_name_without_type}_noised_{duration_seconds}.wav')
    soundfile.write(path, splited_filename, sr)  # Write the truncated audio signal to the temporary WAV file
    return path


def balance_sounds(wav_noise_arr):
    return wav_noise_arr / 4


def return_true_by_probability(probability):
    """
    function which return true with probability as input to help decide if the current audio file will be noisy or not
    """
    return random() < probability


def folder_audio_noiser(probability):
    """
    the function get a wav dir and by probability, copy the file, insert noise and write it in the audio dir
    """

    global directory_path, noise, noise_arr_new, new_noise_path, new_directory_name, parent_directory, parent_directory, parent_directory

    if probability > 1 or probability < 0:
        print(f'ERROR: probability must be between 0 and 1 but the provided probability was {probability}')
        return

    parent_directory = os.path.dirname(wav_dir)  # Get the parent directory of the given path
    parent_directory_name = f'{os.path.basename(wav_dir)}_noised'

    # directory_path = os.path.join(parent_directory, parent_directory_name)

    # if not os.path.exists(os.path.join(directory_path)):
    #     # create the folder which will contain the noisy data
    #     os.mkdir(directory_path)

    directory_path = wav_dir

    counter_all = 0

    noise_path = os.path.join(path_noises,
                              os.listdir(path_noises)[0])  # create the init path to noise(the first in the list)
    noise = os.listdir(path_noises)[0]
    noise_index = 0

    for w_file in os.listdir(wav_dir):
        w_file_path = os.path.join(wav_dir, w_file)
        if not return_true_by_probability(probability):
            audio_arr, sr = librosa.load(w_file_path)
            path_to_noisy_audio = f'{directory_path}\\{w_file}'
            soundfile.write(path_to_noisy_audio, audio_arr, sr)
            continue

        if counter_all == count_dir(wav_dir):
            print(f'End of process: {count_dir(wav_dir)} audio files passed with noise!')

        if get_duration(w_file_path) < get_duration(noise_path):
            new_noise_path = resize_noise(noise_path, int(get_duration(w_file_path)))

        audio_arr, sr1 = librosa.load(w_file_path)  # get the array of numbers which represents the audio
        noise_arr, sr2 = librosa.load(new_noise_path)

        if sr1 != sr2:
            noise_arr_new = librosa.resample(noise_arr, sr2,
                                             sr1)  # Resample audio_arr to the same sample rate as noise_arr

        noise_arr_new = balance_sounds(noise_arr)

        noise_audio = [sum(x) for x in zip(audio_arr, noise_arr_new)]
        split_list = w_file.split("_")
        split_list.insert(len(split_list) - 1, "noise")

        noisy_audio_name = "_".join(split_list)

        path_to_noisy_audio = f'{directory_path}\\{noisy_audio_name}'
        soundfile.write(path_to_noisy_audio, noise_audio, sr1)  # Save the mixed audio data as a new audio file
        print(f'{counter_all}_new file {path_to_noisy_audio} was added with {noise} noise')
        noise, noise_index = get_next_noise(path_noises, noise_index)
        counter_all += 1


def list_folder_noiser(folders, probability):
    """
    input:
        folders: (list) contain the path to the folders to be noisy
        probability: (float) int the range [0,1]
    output:
        the function iterate the folders and make noise in them
    """
    global wav_dir
    for folder_path in folders:
        wav_dir = folder_path
        folder_audio_noiser(probability)
        print(f'{folder_path} was noised successfully!')


if __name__ == '__main__':
    wav_dir = f'.\\..\\data\\training\\Actor_01\\'  # you can replace the relative path to dir which have audio
    path_noises = f'.\\noises'  # you can change the path to a dir which have noise
    folder_audio_noiser(0.2)
    # list_of_folders = [f'.\\..\\data\\training\\Actor_02',
    #                    f'.\\..\\data\\training\\Actor_04',
    #                    f'.\\..\\data\\training\\Actor_03',
    #                    f'.\\..\\data\\training\\Actor_05',
    #                    f'.\\..\\data\\training\\Actor_06',
    #                    f'.\\..\\data\\training\\Actor_07',
    #                    f'.\\..\\data\\training\\Actor_08',
    #                    f'.\\..\\data\\training\\Actor_09',
    #                    f'.\\..\\data\\training\\Actor_10',
    #                    f'.\\..\\data\\training\\Actor_11',
    #                    f'.\\..\\data\\training\\Actor_12',
    #                    f'.\\..\\data\\training\\Actor_13',
    #                    f'.\\..\\data\\training\\Actor_14',
    #                    f'.\\..\\data\\training\\Actor_15',
    #                    f'.\\..\\data\\training\\Actor_16',
    #                    f'.\\..\\data\\training\\Actor_17',
    #                    f'.\\..\\data\\training\\Actor_18',
    #                    f'.\\..\\data\\training\\Actor_19',
    #                    f'.\\..\\data\\training\\Actor_20',
    #                    f'.\\..\\data\\training\\Actor_21',
    #                    f'.\\..\\data\\training\\Actor_22',
    #                    f'.\\..\\data\\training\\Actor_23',
    #                    f'.\\..\\data\\training\\Actor_24',
    #                    f'.\\..\\data\\training\\Actor_25',
    #                    f'.\\..\\data\\training\\Actor_26',
    #                    ]
    # list_folder_noiser(list_of_folders, probability=0.2)