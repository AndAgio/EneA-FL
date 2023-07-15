import os
import pandas as pd

# BASE_DIR = "./reports/"
BASE_DIR = "reports/"
DEVICES = ["cpu/", "gpu/"]
# IOT_MODELS = ["jetson_nano/", "raspberrypi/", "jetson_orin/"]
IOT_MODELS = ["jetson_xavier/"]
#%%
def get_epoch_timestamps(data):
    timestamps = []
    for d in data:
        if not d.startswith("-"):
            timestamps.append(float(d.split(" ")[2]))
    return timestamps[:6]

def get_n_samples_per_epoch(data):
    n_samples = []
    for d in data:
        if not d.startswith("-"):
            n_samples.append(float(d.split(" ")[-1]))
    return n_samples[6:]

def get_couple_timestamps(timestamps):
    couples = []
    couple = []
    for t in timestamps:
        couple.append(t)
        if len(couple) == 2:
            couples.append(couple)
            couple = [couple[1]]
    return couples

def get_sub_df_from_timestamps(df, couple_timestamps):
    dfs_split = []
    for ct in couple_timestamps:
        df_split = df[(df["timestamp"] > ct[0]) & (df["timestamp"] < ct[1])]
        dfs_split.append(df_split)
    return dfs_split

def save_dfs_split(dfs_split, iot_model, device, file_name):
    for i, df in enumerate(dfs_split):
        path = os.path.join(BASE_DIR, iot_model, device, "split_dfs/")
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, str(i) + "_" + file_name.split(".")[0] + ".csv")
        df.to_csv(path, index=False)
#%%
for iot_model in IOT_MODELS:
    for device in DEVICES:
        path = os.path.join(BASE_DIR, iot_model, device)
        files = os.listdir(path)
        for file in files:
            print('Processing file: {}'.format(file))
            file_path = os.path.join(path, file)
            if file.endswith(".txt"):
                with open (file_path, "r") as myfile:
                    data = myfile.read().splitlines()
                timestamps = get_epoch_timestamps(data)
                couple_timestamps = get_couple_timestamps(timestamps)
                this_df = pd.read_csv(file_path.replace(".txt", ".csv"))
                dfs_split = get_sub_df_from_timestamps(this_df, couple_timestamps)
                save_dfs_split(dfs_split, iot_model, device, file)