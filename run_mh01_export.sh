#!/bin/bash

# MH01 Dataset Factor Export Script
# Purpose: Run ORB-SLAM3 to process MH01 dataset and export factor data for hardware modules

set -e  # Exit immediately if a command exits with a non-zero status

# Get script directory (ORB_SLAM3 root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration parameters
VOCABULARY="Vocabulary/ORBvoc.txt"
SETTINGS="Examples/Monocular-Inertial/EuRoC.yaml"
DATASET="dataset/MH01"
TIMESTAMPS="Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt"
OUTPUT_DIR="/home/zlw/End2End/QR/output/mh01"
EXECUTABLE="Examples/Monocular-Inertial/mono_inertial_hw_export"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== MH01 Dataset Factor Export Script ===${NC}"
echo ""

# Check executable
if [ ! -f "$EXECUTABLE" ]; then
    echo -e "${RED}Error: Executable not found: $EXECUTABLE${NC}"
    echo "Please build ORB-SLAM3 first:"
    echo "  cd build && cmake .. && make mono_inertial_hw_export -j\$(nproc)"
    exit 1
fi

# Check input files
echo -e "${YELLOW}Checking input files...${NC}"
missing_files=0

if [ ! -f "$VOCABULARY" ]; then
    echo -e "${RED}  ✗ Vocabulary file not found: $VOCABULARY${NC}"
    missing_files=1
else
    echo -e "${GREEN}  ✓ Vocabulary file: $VOCABULARY${NC}"
fi

if [ ! -f "$SETTINGS" ]; then
    echo -e "${RED}  ✗ Settings file not found: $SETTINGS${NC}"
    missing_files=1
else
    echo -e "${GREEN}  ✓ Settings file: $SETTINGS${NC}"
fi

if [ ! -d "$DATASET" ]; then
    echo -e "${RED}  ✗ Dataset directory not found: $DATASET${NC}"
    missing_files=1
else
    echo -e "${GREEN}  ✓ Dataset directory: $DATASET${NC}"
fi

if [ ! -f "$TIMESTAMPS" ]; then
    echo -e "${RED}  ✗ Timestamp file not found: $TIMESTAMPS${NC}"
    missing_files=1
else
    echo -e "${GREEN}  ✓ Timestamp file: $TIMESTAMPS${NC}"
fi

if [ $missing_files -eq 1 ]; then
    echo -e "${RED}Please prepare missing files first!${NC}"
    exit 1
fi

# Create output directory
echo ""
echo -e "${YELLOW}Preparing output directory...${NC}"
mkdir -p "$OUTPUT_DIR"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}  ✓ Output directory: $OUTPUT_DIR${NC}"
else
    echo -e "${RED}  ✗ Failed to create output directory: $OUTPUT_DIR${NC}"
    exit 1
fi

# Run export program
echo ""
echo -e "${YELLOW}Starting export program...${NC}"
echo "Command: $EXECUTABLE \\"
echo "         $VOCABULARY \\"
echo "         $SETTINGS \\"
echo "         $DATASET \\"
echo "         $TIMESTAMPS \\"
echo "         $OUTPUT_DIR"
echo ""

"$EXECUTABLE" \
    "$VOCABULARY" \
    "$SETTINGS" \
    "$DATASET" \
    "$TIMESTAMPS" \
    "$OUTPUT_DIR"

# Check execution result
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=== Export Completed ===${NC}"
    echo ""
    echo "Output files location: $OUTPUT_DIR"
    echo ""
    if [ -f "$OUTPUT_DIR/poses.json" ]; then
        echo -e "${GREEN}  ✓ poses.json${NC}"
    fi
    if [ -f "$OUTPUT_DIR/camera_observations.json" ]; then
        echo -e "${GREEN}  ✓ camera_observations.json${NC}"
    fi
    if [ -f "$OUTPUT_DIR/imu_edges.json" ]; then
        echo -e "${GREEN}  ✓ imu_edges.json${NC}"
    fi
    if [ -f "$OUTPUT_DIR/priors.json" ]; then
        echo -e "${GREEN}  ✓ priors.json${NC}"
    fi
    if [ -f "$OUTPUT_DIR/summary.json" ]; then
        echo -e "${GREEN}  ✓ summary.json${NC}"
    fi
    echo ""
    echo "Next step: Parse these JSON files and construct CameraObservation / ImuConstraint / PriorConstraint"
else
    echo ""
    echo -e "${RED}=== Export Failed ===${NC}"
    echo "Please check error messages and ensure:"
    echo "  1. MH01 dataset is fully extracted (image files exist)"
    echo "  2. ORB-SLAM3 is properly built"
    echo "  3. All input file paths are correct"
    exit 1
fi
