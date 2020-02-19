source /mtkoss/Modules/3.2.6/x86_64/init/sh
module load Python3/3.6.3_gpu_tf1131
config=$1
echo $PYTHONPATH
python3 ${PYTHONPATH}/representation/netx/main.py --config $config
