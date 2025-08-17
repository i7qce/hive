FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

ENV TERM xterm-256color
ENV TZ=America/New_York

# Below from https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
ARG USERNAME=i7qce
ARG USER_UID=1003
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME 
RUN useradd --uid $USER_UID --gid $USER_GID -m $USERNAME
# [Optional] Add sudo support. Omit if you don't need to install software after connecting.
RUN apt-get update
RUN apt-get install -y sudo
RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME
RUN chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

RUN apt-get update
RUN apt install -y unminimize
RUN yes | unminimize
RUN apt-get install -y ubuntu-standard
RUN apt install -y python3.12-venv
RUN apt install -y python-is-python3
RUN apt install -y python3-dev
RUN apt install -y python3-pip
RUN apt install -y build-essential
RUN apt install -y vim
RUN apt install -y git
RUN apt install -y tmux
RUN apt install -y htop
RUN apt install -y curl
RUN apt install -y golang

# For shaders
RUN apt install -y libglfw3-dev   
RUN apt install -y libglew-dev
RUN apt install -y libstb-dev
RUN apt install -y libglm-dev
RUN apt install -y xvfb

# For opencv
RUN apt install -y libgl1

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

# Copy and apply all dotfiles
COPY .bashrc /opt/configs/.bashrc
COPY .zshrc /opt/configs/.zshrc
COPY .vimrc /opt/configs/.vimrc

RUN cat /opt/configs/.bashrc >> ~/.bashrc
RUN cat /opt/configs/.zshrc >> ~/.zshrc
RUN cat /opt/configs/.vimrc >> ~/.vimrc

# git info
RUN git config --global user.name i7qce
RUN git config --global user.email 111165003+i7qce@users.noreply.github.com

CMD ["bash", "-l"]
