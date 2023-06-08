FROM python:3.11-slim

# setup unix
RUN apt-get update
RUN apt-get install -y \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    libdbus-1-dev \
    libegl-dev \
    libwayland-dev \
    libx11-dev \
    libxcursor-dev \
    libxi-dev \
    libxinerama-dev \
    libxkbcommon-dev \
    libxrandr-dev \
    libxxf86vm-dev \
    linux-libc-dev \
    subversion \
    wayland-protocols \
    wget \
    xz-utils \
    yasm

# install azure cli
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

# install azcopy
RUN wget https://aka.ms/downloadazcopy-v10-linux -O azcopy.tar
RUN mkdir azcopy
RUN tar xvf azcopy.tar -C azcopy --strip-components=1
RUN mv azcopy/azcopy /usr/bin/azcopy
RUN rm -rfd azcopy.tar azcopy

# install blender
ENV BLENDER_URL https://download.blender.org/release/Blender2.90
ENV BLENDER_FNAME blender-2.90.0-linux64
RUN wget $BLENDER_URL/$BLENDER_FNAME.tar.xz
RUN tar xvf $BLENDER_FNAME.tar.xz
RUN rm $BLENDER_FNAME.tar.xz
RUN mv $BLENDER_FNAME /blender

# install aom
RUN git clone https://aomedia.googlesource.com/aom
RUN mkdir aom_build
RUN cd aom_build && cmake ../aom && make && make install

# setup python
RUN pip install --upgrade pip
RUN pip install sh typer
ENV SQLALCHEMY_SILENCE_UBER_WARNING=1

# set working directory
WORKDIR /code

# run batch_task.py as entrypoint
COPY batch_task.py .
ENTRYPOINT ["python", "batch_task.py"]
