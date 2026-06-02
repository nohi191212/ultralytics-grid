export PATH="/opt/conda/bin:$PATH"
conda init

pip install opencv-python matplotlib
pip install -e .

echo "Installation completed!"
