#!/usr/bin/env bash

SHELL=zsh
if [ "$1" = "bash" ]; then
  SHELL=bash
fi

if [ -z $DOCKER_NAME ];then
    DOCKER_NAME="${USER}_mm_docker"
fi

if [ -S $SSH_AUTH_SOCK ]; then
    SSH_AGENT_DIR=/tmp/.ssh-agent-$USER/
    mkdir -p ${SSH_AGENT_DIR}
    ln -f $SSH_AUTH_SOCK ${SSH_AGENT_DIR}/agent.sock 2>/dev/null >/dev/null
fi

xhost +local:root 1>/dev/null 2>&1
docker exec \
    -u $USER \
    -it ${DOCKER_NAME} \
    /bin/$SHELL
xhost -local:root 1>/dev/null 2>&1
