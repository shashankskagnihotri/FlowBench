#!/bin/bash
set -e  # Exit on error

# Configuration
SERVER_URL="https://darus.uni-stuttgart.de"
DEST_DIR="./datasets/adv_weather_data"
PROCESSES=80  # Fixed number of parallel processes

# Function to parallel unzip with progress
parallel_unzip() {
    local zipfile=$1
    
    # Create destination directory if it doesn't exist
    mkdir -p "$DEST_DIR"
    
    # Get total number of files
    local TOTAL_FILES=$(unzip -l "$zipfile" | awk 'END{print NR-4}')
    echo "Extracting $zipfile using $PROCESSES processes..."
    
    # Create a temporary directory for tracking progress
    local TEMP_DIR=$(mktemp -d)
    trap 'rm -rf "$TEMP_DIR"' EXIT
    
    # Start extraction in background and get its PID
    unzip -l "$zipfile" | awk 'NR>3 {print $4}' | head -n -2 | \
    xargs -P "$PROCESSES" -I {} bash -c '
        unzip -q -o "$1" "$2" -d "$3"
        touch "$4/$(date +%s.%N)"
    ' _ "$zipfile" "{}" "$DEST_DIR" "$TEMP_DIR" &
    local UNZIP_PID=$!
    
    # Show progress while the unzip process is running
    while kill -0 $UNZIP_PID 2>/dev/null; do
        local CURRENT=$(ls "$TEMP_DIR" | wc -l)
        local percentage=$((CURRENT * 100 / TOTAL_FILES))
        printf "\rExtracting: [%-50s] %d%%" "$(printf '#%.0s' $(seq 1 $((percentage / 2))))" "$percentage"
        sleep 0.1
    done
    
    # Ensure we show 100% at completion
    printf "\rExtracting: [%-50s] %d%%\n" "$(printf '#%.0s' $(seq 1 50))" 100
    
    # Wait for completion and cleanup
    wait $UNZIP_PID
    echo "Extraction of $zipfile complete!"
    rm "$zipfile"
}

# File IDs and their persistent IDs
declare -A FILE_IDS=(
    ["particles_3000_npz.zip"]="doi:10.18419/DARUS-3677/28"
    ["rain_1500_npz.zip"]="doi:10.18419/DARUS-3677/22"
    ["size_fog_npz.zip"]="doi:10.18419/DARUS-3677/24"
)

# Create a temporary download directory that will be cleaned up
DOWNLOAD_DIR=$(mktemp -d)
trap 'rm -rf "$DOWNLOAD_DIR"' EXIT

# Download each file using its persistent ID
for filename in "${!FILE_IDS[@]}"; do
    persistentId="${FILE_IDS[$filename]}"
    echo "Downloading $filename (ID: $persistentId)..."
    
    # Download to temporary directory
    curl -L -H "X-Dataverse-key:${API_TOKEN:-}" \
        "$SERVER_URL/api/access/datafile/:persistentId/?persistentId=$persistentId" \
        -o "$DOWNLOAD_DIR/$filename"
    
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded $filename"
        # Parallel unzip the downloaded file
        parallel_unzip "$DOWNLOAD_DIR/$filename"
    else
        echo "Failed to download $filename"
        exit 1
    fi
done

echo "Download and extraction completed!"
echo "Data is available in $DEST_DIR"