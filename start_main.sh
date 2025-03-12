#!/bin/bash

VENV_DIR="env"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
fi

os_name=$(uname)

if [ "$os_name" == "Linux" ] || [ "$os_name" == "Darwin" ]; then
    source $VENV_DIR/bin/activate
else
    $VENV_DIR/Scripts/activate.bat
fi

pip install --upgrade pip
pip install opencv-python ultralytics "numpy<2" keyboard

echo ""
echo "All dependencies are installed. Starting the program..."
echo "Press space to view objects detected and 'c' to view colours detected"

sudo python main.py


