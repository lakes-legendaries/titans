#!/bin/bash

# error on failure
set -e

# setup unix
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get dist-upgrade -y
sudo apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    default-libmysqlclient-dev \
    g++ \
    gnupg \
    lsb-release \
    software-properties-common \
    wget \

# access docker repository
KEYFILE=/usr/share/keyrings/docker-archive-keyring.gpg
sudo rm -f $KEYFILE
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o $KEYFILE
echo "deb [arch=$(dpkg --print-architecture) signed-by=$KEYFILE] \
    https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# install docker engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# install azure cli and azcopy
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
wget https://aka.ms/downloadazcopy-v10-linux -O azcopy.tar
mkdir azcopy
tar xvf azcopy.tar -C azcopy --strip-components=1
sudo mv azcopy/azcopy /usr/bin/azcopy
rm -rfd azcopy.tar azcopy

# get ssl/tls certificates for secure https connection
sudo apt-get install -y snapd
sudo snap install core
sudo snap refresh core
sudo apt-get remove -y certbot
sudo snap install --classic certbot
sudo ln --force -s /snap/bin/certbot /usr/bin/certbot
sudo /usr/bin/certbot certonly \
    --standalone -n --domains titansapi.eastus.cloudapp.azure.com \
    --agree-tos --email mike@lakeslegendaries.com

# upgrade python
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get install -y python3.11 python3.11-dev python3.11-venv

# set up basic aliases and environmental variables
echo "alias python=python3.11" > ~/.bash_aliases
echo "alias venv=\"source .venv/bin/activate\"" >> ~/.bash_aliases
echo "export PYTHONPATH=\".:/home/mike/titans\"" >> ~/.bash_aliases
echo "export SECRETS_DIR=\"/home/mike/secrets\"" >> ~/.bash_aliases
echo "export TF_CPP_MIN_LOG_LEVEL=3" >> ~/.bash_aliases

# schedule restart and daily updates
CRONDIR="/var/spool/cron/crontabs"
URL=https://raw.githubusercontent.com/lakes-legendaries/titans/main/webserver/run-service.sh
sudo rm -f $CRONDIR/$USER
sudo rm -f $CRONDIR/root
echo "@reboot /bin/bash -c \"curl $URL | bash\"" | sudo tee -a $CRONDIR/$USER
echo "0 0 1 * * /sbin/shutdown -r now" | sudo tee $CRONDIR/root
sudo chmod 0600 $CRONDIR/$USER
sudo chmod 0600 $CRONDIR/root

# reboot
sudo reboot
