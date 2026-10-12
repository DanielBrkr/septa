#FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime
FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime
#FROM ubuntu:24.04
#FROM python:3.12.11-slim-bookworm
LABEL authors="danielbreucker"

# this has to be aligned with the user on the host machine due to permissions (related to podman)
ENV CONTAINER_USER="breucker"
ENV CONTAINER_USER_HOST_UID="2072"
ENV CONTAINER_USER_HOST_GID="1111"
ENV PROJECT_DIR="digital-icu-deterioration-prediction"

WORKDIR /home/${CONTAINER_USER}

# some developer tooling (opinionated)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    openssh-server \
    mc \
    graphviz \
    nvtop \
    git \
    curl \
    gpg \
    zsh \
    neovim \
    tmux \
    language-pack-en \
    sudo \
    && apt-get clean

# Sets the credentials for the SSH connection
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config \
    && echo "PermitEmptyPasswords yes" >> /etc/ssh/sshd_config \
    && echo "AllowTcpForwarding yes" >> /etc/ssh/sshd_config \
    && echo "root:dev" | chpasswd

# SSH Login Fix for containers
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

# this doesn't do anything apart, from indicating that this port has to be exposed
EXPOSE 22

# --- User Setup
# Adds a new user space to the container
RUN groupadd -g ${CONTAINER_USER_HOST_GID}  ${CONTAINER_USER} \
    && useradd -m ${CONTAINER_USER} -u ${CONTAINER_USER_HOST_UID} -g ${CONTAINER_USER_HOST_GID}  \
    && passwd -d ${CONTAINER_USER}

# Allows the user sudo rights in the container
RUN echo "${CONTAINER_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN chown ${CONTAINER_USER} /home/${CONTAINER_USER}
RUN chsh -s /bin/bash ${CONTAINER_USER}


# --- Dev Tooling Setup
# this suspect to developer preference, you may want to change that to your needs
ENV RUNNING_IN_DOCKER true

# use with docker exec -it CONTAINER_NAME bin/zsh
RUN sudo -u ${CONTAINER_USER} sh -c "$(wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"
RUN sudo -u ${CONTAINER_USER} sed -i 's/ZSH_THEME=".*"/ZSH_THEME="agnoster"/' /home/${CONTAINER_USER}/.zshrc && \
    locale-gen en_US.UTF-8 # fix for anon (12): character not in range

WORKDIR /home/${CONTAINER_USER}/${PROJECT_DIR}

COPY startup_script.sh /home/${CONTAINER_USER}/${PROJECT_DIR}
COPY pyproject.toml /home/${CONTAINER_USER}/${PROJECT_DIR}
COPY uv.lock /home/${CONTAINER_USER}/${PROJECT_DIR}

# This is for useful for debuggin
#RUN pwd
#RUN whoami

# --- uv setup
# prevents permission error related to the python interpreter (due to default location)
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python \
#    UV_PROJECT_ENVIRONMENT=/home/${CONTAINER_USER}/${PROJECT_DIR}/.venv \
    UV_PROJECT_ENVIRONMENT=/opt/.venv \
    UV_LINK_MODE=copy \
#    PATH="/home/${CONTAINER_USER}/${PROJECT_DIR}/.venv/bin:$PATH" \
    PATH="/opt/.venv/bin:$PATH" \
    PYTHONPATH="${PYTHONPATH}:/home/${CONTAINER_USER}/${PROJECT_DIR}" \
#    PYTHONPATH="${PYTHONPATH}:/opt/.venv" \
    PATH="/root/.local/bin/:$PATH"

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /bin/

# without caching
#RUN uv sync --frozen # FIXME: Doesn't seem to work

# with caching
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

RUN chown ${CONTAINER_USER} /home/${CONTAINER_USER}/${PROJECT_DIR}

USER root

RUN mkdir /run/sshd && \
    ssh-keygen -A

# podman needs this and entry point it seems, there's a crun permission denied error otherwise
# you need to run chmod +x/or chmod 755 additionally on the host machine!
RUN chmod +x ./startup_script.sh

ENTRYPOINT ["./startup_script.sh"]
