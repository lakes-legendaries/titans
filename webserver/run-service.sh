#!/bin/bash

# exit on error
set -e

# renew certificates, copy into secrets
sudo certbot renew
for FILE in \
    /etc/letsencrypt/live/titansapi.eastus.cloudapp.azure.com/fullchain.pem \
    /etc/letsencrypt/live/titansapi.eastus.cloudapp.azure.com/privkey.pem \
; do
    sudo cp $FILE ~/secrets/
done

# remove existing docker containers and images
CONTAINERS="$(sudo docker ps -q --filter publish=443)"
if [ ! -z "$CONTAINERS" ]; then
    sudo docker rm --force "$CONTAINERS"
fi
sudo docker system prune --force --all

# clone repo
API_DIR=~/api
rm -rfd $API_DIR
git clone https://github.com/lakes-legendaries/titans $API_DIR
cd $API_DIR

# rebuild docker image
sudo docker build -t titans . --no-cache

# start api service
sudo docker run -dp 443:443 -v ~/secrets:/secrets titans

# clean up
rm -rfd $API_DIR
