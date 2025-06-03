import os
import shutil

# Define the main function
def organize_pdfs(src_directory):
    # List all the PDF files in the source directory
    pdf_files = [f for f in os.listdir(src_directory) if f.endswith('.pdf')]
    
    # Iterate over each PDF file
    for pdf_file in pdf_files:
        # Split the filename at the first underscore
        prefix = pdf_file.split('_')[0]

        # For "3dcc" and "common" prefixes, use special folder names
        if prefix in ['3dcc', 'common']:
            # Determine which special folder to use
            if 'sintel_clean' in pdf_file:
                subfolder = 'sintel_clean'
            elif 'sintel_final' in pdf_file:
                subfolder = 'sintel_final'
            elif 'kitti' in pdf_file:
                subfolder = 'kitti'
            else:
                print(f"Warning: '{pdf_file}' does not contain 'sintel_clean', 'sintel_final', or 'kitti'. Skipping.")
                continue

        # For other prefixes, continue with 'inf' and 'two'
        else:
            if 'inf' in pdf_file:
                subfolder = 'inf'
            elif 'two' in pdf_file:
                subfolder = 'two'
            else:
                print(f"Warning: '{pdf_file}' does not contain 'inf' or 'two'. Skipping.")
                continue

        # Create the directory structure based on the prefix
        prefix_dir = os.path.join(src_directory, prefix)
        inf_or_two_or_special_dir = os.path.join(prefix_dir, subfolder)
        all_models_dir = os.path.join(inf_or_two_or_special_dir, 'all_models')
        top_10_models_dir = os.path.join(inf_or_two_or_special_dir, 'top_10_models')

        # Create the directories if they don't exist
        os.makedirs(all_models_dir, exist_ok=True)
        os.makedirs(top_10_models_dir, exist_ok=True)

        # Determine where to copy the PDF based on whether 'top_10' is in the filename
        if 'top_10' in pdf_file:
            target_dir = top_10_models_dir
        else:
            target_dir = all_models_dir

        # Copy the PDF to the target directory
        src_path = os.path.join(src_directory, pdf_file)
        dst_path = os.path.join(target_dir, pdf_file)
        shutil.copy(src_path, dst_path)
        print(f"Copied '{pdf_file}' to '{target_dir}'.")

if __name__ == "__main__":
    # Specify the source directory where the PDF files are located
    src_directory = './'
    
    # Call the function to organize PDFs
    organize_pdfs(src_directory)
