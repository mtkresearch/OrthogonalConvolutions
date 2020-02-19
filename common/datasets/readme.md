# On the local machine
Original datasets should be stored in the non-local directory /data/raw/.  
The /data directory is automatically mounted to the docker container.
Datasets must be processed into h5 format using scripts in /code/common/datasets/process, afterwhich it can be used for deep learning by pointing to the path.

# On the farm
Original datasets located in /site/mtkdatasets/Public

# Downloading to local machine and processing it
## Cifar10
```
mkdir -p /data/raw
cd /data/raw
sudo wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
sudo tar -xvf cifar-10-binary.tar.gz

```
To process the binary files
```
mkdir -b /data/cifar
cd ~/code/common/dataset/process
python cifar10.py
```

## Cifar100
```
mkdir -p /data/raw
cd /data/raw
sudo wget https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
sudo tar -xvf cifar-100-binary.tar.gz

```
