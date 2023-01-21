FROM python:3.9-slim

# setup unix
RUN apt-get update
RUN apt-get install -y curl wget xz-utils

# install azure cli
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

# install blender
ENV BLENDER_URL https://download.blender.org/release/Blender2.90
ENV BLENDER_FNAME blender-2.90.0-linux64
RUN wget $BLENDER_URL/$BLENDER_FNAME.tar.xz
RUN tar xvf $BLENDER_FNAME.tar.xz
RUN rm $BLENDER_FNAME.tar.xz
RUN mv $BLENDER_FNAME /blender

# set working directory
WORKDIR /code

# copy in blender files
COPY assets /code/assets
COPY blend /code/blend
