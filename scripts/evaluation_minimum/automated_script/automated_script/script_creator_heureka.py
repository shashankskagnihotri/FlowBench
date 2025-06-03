import os
import shutil

# Define your lists
dataset_names = ["kitti-2015", "sintel-clean", "sintel-final"]
kitti_model_names = [
    "raft", "gma", "rpknet", "ccmr", "craft", "csflow", "dicl", "dip",
    "fastflownet", "maskflownet", "flow1d", "flowformer", "flowformer++", 
    "gmflow", "gmflownet", "hd3", "irr_pwc", "liteflownet",
    "liteflownet3_pseudoreg", "llaflow", "matchflow", "ms_raft+", 
    "rapidflow", "scopeflow", "scv4", "separableflow", "skflow", "starflow", 
    "videoflow_bof"
]
sintel_model_names = [
    "raft", "pwcnet", "gma", "rpknet", "ccmr", "craft", "dicl", "dip", 
    "fastflownet", "maskflownet", "maskflownet_s", "flow1d", "flowformer", 
    "flowformer++", "gmflow", "hd3", "irr_pwc", "liteflownet", "liteflownet2", 
    "liteflownet3", "llaflow", "matchflow", "ms_raft+", 
    "rapidflow", "scopeflow", "scv4", "separableflow", "skflow", "starflow", 
    "videoflow_bof"
]
gpu_a100_models = ["ccmr", "flowformer", "flowformer++", "dip", "ms_raft+"]
# Change directory to 2 levels below the current one
os.chdir(os.path.join(os.getcwd(), "..", ".."))

# Get all folder names in the current directory
folders = [f for f in os.listdir() if os.path.isdir(f) and f != "automated_script"]

# Iterate through the folders
for folder in folders:
    if folder in kitti_model_names:
        # Create kitti-2015 directory if it does not exist
        kitti_dir = os.path.join(folder, "kitti-2015")
        os.makedirs(kitti_dir, exist_ok=True)

        # Copy script files from automated_script/kitti-2015
        source_dir = os.path.join("automated_script", "kitti-2015")
        for file in os.listdir(source_dir):
            if file.endswith("heureka.sh"):  # Only process files ending with 'heureka.sh'
                src_file = os.path.join(source_dir, file)
                dst_file = os.path.join(kitti_dir, file)

                # Delete destination file if it already exists
                if os.path.exists(dst_file):
                    os.remove(dst_file)

                shutil.copy2(src_file, dst_file)

                # Edit the script files
                with open(dst_file, "r") as file:
                    data = file.read()
                data = data.replace("model_name", folder)
                if folder in gpu_a100_models:
                    data = data.replace("accelerated", "accelerated-h100")
                if folder == "ms_raft+":
                    data = data.replace('checkpoint="kitti"', 'checkpoint="mixed"')

                with open(dst_file, "w") as file:
                    file.write(data)

                print(f"File created: {dst_file}")

    if folder in sintel_model_names:
        # Create sintel-clean and sintel-final directories if they do not exist
        sintel_clean_dir = os.path.join(folder, "sintel-clean")
        sintel_final_dir = os.path.join(folder, "sintel-final")
        os.makedirs(sintel_clean_dir, exist_ok=True)
        os.makedirs(sintel_final_dir, exist_ok=True)

        # Copy script files from automated_script/sintel-clean
        source_dir = os.path.join("automated_script", "sintel-clean")
        for file in os.listdir(source_dir):
            if file.endswith("heureka.sh"):  # Only process files ending with 'heureka.sh'
                src_file = os.path.join(source_dir, file)
                dst_file = os.path.join(sintel_clean_dir, file)

                # Delete destination file if it already exists
                if os.path.exists(dst_file):
                    os.remove(dst_file)

                shutil.copy2(src_file, dst_file)

                # Edit the script files
                with open(dst_file, "r") as file:
                    data = file.read()
                data = data.replace("model_name", folder)
                if folder in gpu_a100_models:
                    data = data.replace("accelerated", "accelerated-h100")
                if folder == "ms_raft+":
                    data = data.replace('checkpoint="sintel"', 'checkpoint="mixed"')
                with open(dst_file, "w") as file:
                    file.write(data)

                print(f"File created: {dst_file}")

        # Copy script files from automated_script/sintel-final
        source_dir = os.path.join("automated_script", "sintel-final")
        for file in os.listdir(source_dir):
            if file.endswith("heureka.sh"):  # Only process files ending with 'heureka.sh'
                src_file = os.path.join(source_dir, file)
                dst_file = os.path.join(sintel_final_dir, file)

                # Delete destination file if it already exists
                if os.path.exists(dst_file):
                    os.remove(dst_file)

                shutil.copy2(src_file, dst_file)

                # Edit the script files
                with open(dst_file, "r") as file:
                    data = file.read()
                data = data.replace("model_name", folder)
                if folder in gpu_a100_models:
                    data = data.replace("accelerated", "accelerated-h100")
                with open(dst_file, "w") as file:
                    file.write(data)
                if folder == "ms_raft+":
                    data = data.replace('checkpoint="sintel"', 'checkpoint="mixed"')

                print(f"File created: {dst_file}")

print("Script execution completed.")
