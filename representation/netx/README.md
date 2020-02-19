# Set the python path
export PYTHONPATH=/code

# Use launch.py to launch jobs on the farm
python3 launch.py --config config.json --n_gpus 1 --gpu_type GPU_1080

# To run the main programing
python3 main.py --config config.json
