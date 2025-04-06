import os
import gdown

def download_model():
    if not os.path.exists("flood_model.pkl"):
        file_id = "1QO_bbBWd9iHStWl6kGkPwExFm1cmY1Hk"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, "flood_model.pkl", quiet=False)
