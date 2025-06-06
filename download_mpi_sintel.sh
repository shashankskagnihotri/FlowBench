#!/bin/bash

# Script to download and unpack MPI Sintel dataset and depth data

# Create a directory for the dataset
mkdir -p datasets/Sintel
cd datasets/Sintel

echo "MPI Sintel Dataset Downloader"
echo "============================"
echo "This script will download:"
echo "- Complete optical flow dataset (~5.3GB)"
echo "- Depth data (~1.5GB)"
echo ""
echo "Total download size will be approximately 6.8GB"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Check if wget is available, otherwise try curl
download_cmd=""
if command -v wget &> /dev/null; then
    download_cmd="wget"
else
    if command -v curl &> /dev/null; then
        download_cmd="curl -L -O"
    else
        echo "Error: Neither wget nor curl is installed. Please install one of them and try again."
        exit 1
    fi
fi

download_and_unpack() {
    local url=$1
    local filename=$2
    local description=$3

    echo "Downloading $description from $url..."

    if [ "$download_cmd" = "wget" ]; then
        wget -O "$filename" "$url"
    else
        curl -L -o "$filename" "$url"
    fi

    if [ $? -eq 0 ]; then
        echo "Download complete. Unpacking..."
        unzip -o "$filename"
        echo "Unpacking $description complete."
    else
        echo "Error downloading $description"
        return 1
    fi
}

# Main MPI Sintel Dataset (Complete)
echo "Step 1/2: Downloading MPI Sintel Complete Dataset"
download_and_unpack "http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip" "MPI-Sintel-complete.zip" "complete dataset"

# Depth data
echo "Step 2/2: Downloading MPI Sintel Depth Data"
# For depth data, the exact download URL isn't provided on the webpage, so using a possible URL pattern
# Note: This URL might need to be adjusted based on the actual download link
download_and_unpack "http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-depth-training-20150305.zip" "MPI-Sintel-depth.zip" "depth data"

# Check if any downloads failed
check_all_downloads() {
    if [ ! -f "MPI-Sintel-complete.zip" ] || [ ! -f "MPI-Sintel-depth.zip" ]; then
        echo "Some downloads may have failed."
        echo "Note: The URL for the depth data is guessed and might not be accurate."
        echo "You might need to manually download the depth dataset from:"
        echo "- Depth data: http://sintel.is.tue.mpg.de/depth"
        echo "This page may require registration or have a different download URL."
    else
        echo "All downloads completed successfully!"
    fi
}

# Optional: Remove zip files to save space
cleanup() {
    read -p "Do you want to remove the zip files to save space? (y/n): " choice
    if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
        echo "Removing zip files..."
        rm -f MPI-Sintel-complete.zip MPI-Sintel-depth.zip
        echo "Zip files removed."
    fi
}

check_all_downloads
cleanup

echo "Done."
