import glob
import pandas as pd
import os


# def write_emodb_csv(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'], train_name="train_emo.csv",
#                     test_name="test_emo.csv", train_size=0.8, verbose=1):
emotions = ["neutral", "calm", "happy", "sad", "angry", "fear", "disgust", "surprised", "ps", "boredom", "exited"]
def write_emodb_csv(emotions=emotions, train_name="train_emo.csv",
                    test_name="test_emo.csv", train_size=0.8, verbose=1):
    """
    Reads speech emodb dataset from directory and write it to a metadata CSV file.
    params:
        emotions (list): list of emotions to read from the folder, default is ['sad', 'neutral', 'happy']
        train_name (str): the output csv filename for training data, default is 'train_emo.csv'
        test_name (str): the output csv filename for testing data, default is 'test_emo.csv'
        train_size (float): the ratio of splitting training data, default is 0.8 (80% Training data and 20% testing data)
        verbose (int/bool): verbositiy level, 0 for silence, 1 for info, default is 1
    """
    target = {"path": [], "emotion": []}
    categories = {
        "W": "angry",
        "L": "boredom",
        "E": "disgust",
        "A": "fear",
        "F": "happy",
        "T": "sad",
        "N": "neutral",
        "C": "calm",
        "J": "exited",
        "S": "surprised",
        "P": "ps"
    }
    # delete not specified emotions
    categories_reversed = { v: k for k, v in categories.items() }
    for emotion, code in categories_reversed.items():
        if emotion not in emotions:
            del categories[code]
        for i, file in enumerate(glob.glob(f"data/emodb/wav/*_{emotion}.wav")):
            # try:
            #     emotion = categories[os.path.basename(file)[5]]
            # except KeyError:
            #     continue
            target['emotion'].append(distributeEmotion(emotion))
            target['path'].append(file)
    if verbose:
        print("[EMO-DB] Total files to write:", len(target['path']))
        
    # dividing training/testing sets
    n_samples = len(target['path'])
    test_size = int((1-train_size) * n_samples)
    train_size = int(train_size * n_samples)
    if verbose:
        print("[EMO-DB] Training samples:", train_size)
        print("[EMO-DB] Testing samples:", test_size)
    X_train, X_test, y_train, y_test = [], [], [], []
    temp_size = 0
    size = int(train_size / 7)
    s = size
    test_s = int(test_size / 7)
    for i in range(1, 10):
        # print(f'size: {temp_size} -> {size}')
        # print(f'test size: {size} -> {size + test_s}')
        X_train += target['path'][temp_size:size]
        X_test += target['path'][size:size+test_s]
        y_train += target['emotion'][temp_size:size]
        y_test += target['emotion'][size:size+test_s]
        temp_size = size + test_s
        size += s
    # print(len(X_train))
    # print(len(X_test))
    pd.DataFrame({"path": X_train, "emotion": y_train}).to_csv(train_name)
    pd.DataFrame({"path": X_test, "emotion": y_test}).to_csv(test_name)


def write_tess_ravdess_csv(emotions=emotions, train_name="train_tess_ravdess.csv",
                            test_name="test_tess_ravdess.csv", verbose=1):
    """
    Reads speech TESS & RAVDESS datasets from directory and write it to a metadata CSV file.
    params:
        emotions (list): list of emotions to read from the folder, default is ['sad', 'neutral', 'happy']
        train_name (str): the output csv filename for training data, default is 'train_tess_ravdess.csv'
        test_name (str): the output csv filename for testing data, default is 'test_tess_ravdess.csv'
        verbose (int/bool): verbositiy level, 0 for silence, 1 for info, default is 1
    """
    train_target = {"path": [], "emotion": []}
    test_target = {"path": [], "emotion": []}
    
    for category in emotions:
        # for training speech directory
        total_files = glob.glob(f"data/training/Actor_*/*_{category}.wav")
        for i, path in enumerate(total_files):
            train_target["path"].append(path)
            train_target["emotion"].append(distributeEmotion(category))
        if verbose and total_files:
            print(f"[TESS&RAVDESS] There are {len(total_files)} training audio files for category:{category}")
    
        # for validation speech directory
        total_files = glob.glob(f"data/validation/Actor_*/*_{category}.wav")
        for i, path in enumerate(total_files):
            test_target["path"].append(path)
            test_target["emotion"].append(distributeEmotion(category))
        if verbose and total_files:
            print(f"[TESS&RAVDESS] There are {len(total_files)} testing audio files for category:{category}")
    pd.DataFrame(test_target).to_csv(test_name)
    pd.DataFrame(train_target).to_csv(train_name)

POSITIVE = 0
NEUTRAL = 1
NEGATIVE = 2
def distributeEmotion(emotion):

    if isinstance(emotion, str):
      emotion = emotion.lower()

    if emotion in {'angry', 'disgust', 'fear', 'sad'}:
      return 'NEGATIVE'

    if emotion in {'neutral', 'calm', 'boredom'}:
      return 'NEUTRAL'

    if emotion in {'happy', 'ps', 'surprised', 'exited'}:
      return 'POSITIVE'

    return -1

def write_custom_csv(emotions=emotions, train_name="train_custom.csv", test_name="test_custom.csv",
                    verbose=1):
    """
    Reads Custom Audio data from data/*-custom and then writes description files (csv)
    params:
        emotions (list): list of emotions to read from the folder, default is ['sad', 'neutral', 'happy']
        train_name (str): the output csv filename for training data, default is 'train_custom.csv'
        test_name (str): the output csv filename for testing data, default is 'test_custom.csv'
        verbose (int/bool): verbositiy level, 0 for silence, 1 for info, default is 1
    """
    train_target = {"path": [], "emotion": []}
    test_target = {"path": [], "emotion": []}
    for category in emotions:

        # train data
        for i, file in enumerate(glob.glob(f"data/train-custom/*_{category}.wav")):
            train_target["path"].append(file)
            # train_target["emotion"].append(category)
            train_target["emotion"].append(distributeEmotion(category))
        if verbose:
            try:
                print(f"[Custom Dataset] There are {i} training audio files for category:{category}")
            except NameError:
                # in case {i} doesn't exist
                pass
        
        # test data
        for i, file in enumerate(glob.glob(f"data/test-custom/*_{category}.wav")):
            test_target["path"].append(file)
            # test_target["emotion"].append(category)
            test_target["emotion"].append(distributeEmotion(category))
        if verbose:
            try:
                print(f"[Custom Dataset] There are {i} testing audio files for category:{category}")
            except NameError:
                pass
    
    # write CSVs
    if train_target["path"]:
        pd.DataFrame(train_target).to_csv(train_name)

    if test_target["path"]:
        pd.DataFrame(test_target).to_csv(test_name)
