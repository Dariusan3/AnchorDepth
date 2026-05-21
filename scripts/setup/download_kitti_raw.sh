#!/bin/bash
# Download KITTI raw dataset sequences needed for Eigen split training/evaluation.

KITTI_DIR="datasets/kitti_raw"
SPLIT_FILE="splits/kitti_all_sequences.txt"
BASE_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"

mkdir -p "$KITTI_DIR"

downloaded=0
skipped=0
failed=0

while IFS= read -r line; do
    date=$(echo "$line" | cut -d'/' -f1)
    drive=$(echo "$line" | cut -d'/' -f2)

    # Download calibration files for this date (once)
    calib_dir="$KITTI_DIR/$date"
    if [ ! -f "$calib_dir/calib_cam_to_cam.txt" ]; then
        echo "Downloading calibration for $date..."
        url="${BASE_URL}/${date}_calib.zip"
        if wget -q -O "/tmp/${date}_calib.zip" "$url"; then
            unzip -q -o "/tmp/${date}_calib.zip" -d "$KITTI_DIR/"
            rm -f "/tmp/${date}_calib.zip"
            echo "  -> Calibration OK"
        else
            echo "  -> FAILED calibration for $date"
            rm -f "/tmp/${date}_calib.zip"
        fi
    fi

    # Skip if already downloaded
    drive_dir="$KITTI_DIR/$date/$drive"
    if [ -d "$drive_dir/image_02/data" ]; then
        n_frames=$(ls "$drive_dir/image_02/data/" | wc -l)
        if [ "$n_frames" -gt 0 ]; then
            echo "SKIP $drive ($n_frames frames)"
            skipped=$((skipped + 1))
            continue
        fi
    fi

    # Download drive
    drive_nosync="${drive%_sync}"
    zip_name="${drive}.zip"
    url="${BASE_URL}/${drive_nosync}/${zip_name}"

    echo "Downloading $drive..."
    if wget -q --show-progress -O "/tmp/$zip_name" "$url"; then
        unzip -q -o "/tmp/$zip_name" -d "$KITTI_DIR/"
        rm -f "/tmp/$zip_name"
        n_frames=$(ls "$drive_dir/image_02/data/" 2>/dev/null | wc -l)
        echo "  -> OK ($n_frames frames)"
        downloaded=$((downloaded + 1))
    else
        echo "  -> FAILED $drive"
        rm -f "/tmp/$zip_name"
        failed=$((failed + 1))
    fi

done < "$SPLIT_FILE"

echo ""
echo "Done! Downloaded: $downloaded, Skipped: $skipped, Failed: $failed"
echo "Total size: $(du -sh $KITTI_DIR | cut -f1)"
echo "Sequences with images: $(find $KITTI_DIR -name 'image_02' -type d | wc -l)"
