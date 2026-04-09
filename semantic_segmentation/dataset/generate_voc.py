import os
import zipfile
import urllib.request
import opendatasets as od

def build_voc_dataset():
    # 1. URLs and Paths
    kaggle_url = "https://www.kaggle.com/datasets/vijayabhaskar96/pascal-voc-2007-and-2012"
    dropbox_url = "https://www.dropbox.com/scl/fi/xccys1fus0utdioi7nj4d/SegmentationClassAug.zip?rlkey=0wl8iz6sc40b3qf6nidun4rez&dl=1"
    
    base_dir = "pascal-voc-2007-and-2012"
    voc_2012_path = os.path.join(base_dir, "VOCdevkit", "VOC2012")
    aug_zip_name = "SegmentationClassAug.zip"

    # 2. Download Kaggle Dataset
    print("\n--- Step 1: Downloading Kaggle Dataset ---")
    # This will ask for your Kaggle Username and Key if not already configured
    od.download(kaggle_url)

    # 3. Download Dropbox Augmentation
    print("\n--- Step 2: Downloading Augmented Labels ---")
    if not os.path.exists(aug_zip_name):
        urllib.request.urlretrieve(dropbox_url, aug_zip_name)
        print("Download complete.")
    else:
        print("Augmented zip already exists, skipping download.")

    # 4. Extract Augmentation into the VOC2012 folder
    print("\n--- Step 3: Merging Augmentation ---")
    if os.path.exists(voc_2012_path):
        with zipfile.ZipFile(aug_zip_name, 'r') as zip_ref:
            # Most VOC zips contain the folder name itself, 
            # so we extract to the parent VOC2012 directory
            zip_ref.extractall(voc_2012_path)
        print(f"Extracted to {voc_2012_path}")
    else:
        print(f"Error: Could not find VOC2012 path at {voc_2012_path}")
        return

    # 5. Verification
    final_aug_path = os.path.join(voc_2012_path, "SegmentationClassAug")
    if os.path.exists(final_aug_path):
        file_count = len([f for f in os.listdir(final_aug_path) if f.endswith('.png')])
        print(f"\n--- Final Verification ---")
        print(f"Path: {final_aug_path}")
        print(f"Files found: {file_count}")
        
        if file_count == 12031:
            print("Status: SUCCESS. Count matches 12,031.")
        else:
            print(f"Status: WARNING. Expected 12,031, but found {file_count}.")
    
    # Optional: Cleanup the zip
    # os.remove(aug_zip_name)

if __name__ == "__main__":
    build_voc_dataset()
