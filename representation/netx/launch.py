#!/usr/bin/python3
import argparse
import json
import os

if __name__ == '__main__':
    # Two step parser first from CLI and the second from config file
    cli_parser = argparse.ArgumentParser(description='Run jobs the GPU farm.')
    
    # Add all the various arguments that the user should be prompted 
    cli_parser.add_argument('--queue', help='Select GPU or CPU queue', default='ML_GPU')
    cli_parser.add_argument('--acc_id', help='Accounting ID', default='d_98001')
    cli_parser.add_argument('--library', help='Library to use: TensorFlow, Pytorch', default='TensorFlow')
    cli_parser.add_argument('--gpu_type', help='Kind of GPU to use: GPU_1080, GPU_2080, GPU_p100', default='GPU_1080')
    cli_parser.add_argument('--n_gpus', type=int, help='Number of GPUs', default=1)
    cli_parser.add_argument('--config', help='Config file to use')
    args = cli_parser.parse_args()

    with open(args.config, 'r') as config_file:
        config_str= config_file.read()

    command = 'bsub -q %s -app %s -P %s -gpu num=%i -m %s sh job.sh \'%s\'' % (args.queue, args.library,
        args.acc_id, args.n_gpus, args.gpu_type, config_str)

    print(os.environ['PYTHONPATH'])
    os.system(command) 
