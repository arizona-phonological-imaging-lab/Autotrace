#!/bin/sh

set -e

apt-get -y install python-dev python-pip g++ \
    libopenblas-dev libhdf5-dev

# some day ubuntu will release a working CUDA repo
# until that day comes, we need to get it straight from nvidia
if [ -e 'cuda-repo-ubuntu1404-7-0-local_7.0-28_amd64.deb' ]; then
    wget 'http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/rpmdeb/cuda-repo-ubuntu1404-7-0-local_7.0-28_amd64.deb'
    dpkg -i 'cuda-repo-ubuntu1404-7-0-local_7.0-28_amd64.deb'
    apt-get update
    apt-get -y install cuda
    echo 'export PATH="$PATH:/usr/local/cuda/bin"' > '/etc/profile.d/cuda.sh'
    echo '/usr/local/cuda/lib64/' > '/etc/ld.so.conf.d/cuda.conf'
    ldconfig
else
    dpkg -i 'cuda-repo-ubuntu1404-7-0-local_7.0-28_amd64.deb'
    apt-get update
    apt-get -y install cuda
    echo 'export PATH="$PATH:/usr/local/cuda/bin"' > '/etc/profile.d/cuda.sh'
    echo '/usr/local/cuda/lib64/' > '/etc/ld.so.conf.d/cuda.conf'
    rm -f 'cuda-repo-ubuntu1404-7-0-local_7.0-28_amd64.deb'
    ldconfig
fi

pip install -r requirements.txt

echo Autotrace dependencies successfully installed.
echo   You may have to reboot before GPU accelleration will work.
