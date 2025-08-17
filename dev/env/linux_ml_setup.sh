# Install torch
echo "Installing torch w/ CUDA"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

echo "Installing lightning"
pip install lightning

# Install more generic libraries
echo "Installing CV libraries"
pip install -r ./requirements_cv.txt