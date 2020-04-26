#!/usr/bin/env bash

SHELL_PATH="$( cd "$( dirname "$0"  )" && pwd  )"

DOCKER_HOME="/home/$USER"
if [ "$USER" == "root" ];then
    DOCKER_HOME="/root"
fi

if [ -z $DOCKER_NAME ];then
    DOCKER_NAME="${USER}_mm_docker"
fi

DOCKER_REPO="docker.fabu.ai:5000/ningqingqun/pytorch"
VERSION="vision-20190730_2219"
IMG=${DOCKER_REPO}:$VERSION
LOCAL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

USER_ID=$(id -u)
GRP=$(id -g -n)
GRP_ID=$(id -g)

function local_volumes() {
  volumes="-v $LOCAL_DIR:/work \
           -v $HOME/.ssh:${DOCKER_HOME}/.ssh \
           -v $HOME/.zsh_history:${DOCKER_HOME}/.zsh_history \
		   -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		   -v /media:/media \
		   -v /etc/localtime:/etc/localtime:ro \
		   -v /private:/private \
		   -v /onboard_data:/onboard_data \
		   -v /nfs:/nfs \
		   -v ${HOME}/.torch:${DOCKER_HOME}/.torch \
		   -v ${HOME}/.cache:${DOCKER_HOME}/.cache \
		   -v /data:/data"

  echo "${volumes}"
}

function add_user() {
  add_script="addgroup --gid ${GRP_ID} ${GRP} && \
      adduser --disabled-password --gecos '' ${USER} \
        --uid ${USER_ID} --gid ${GRP_ID} 2>/dev/null && \
      usermod -aG sudo ${USER} && \
      echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
      cp -r /etc/skel/. /home/${USER} && \
      chsh -s /usr/bin/zsh ${USER} && \
      chown -R ${USER}:${GRP} '/home/${USER}'"
  echo "${add_script}"
}

function config_zsh() {
  config_script="cp -r /oh-my-zsh ${DOCKER_HOME}/.oh-my-zsh && \
      cp ${DOCKER_HOME}/.oh-my-zsh/templates/zshrc.zsh-template ${DOCKER_HOME}/.zshrc && \
      sed -i 's/\"robbyrussell/\"candy/g' ${DOCKER_HOME}/.zshrc"
  echo "${config_script}"
}

function config_pip() {
  config_script="echo '[easy_install]' > '$HOME/.pydistutils.cfg' && \
      echo 'index_url = https://pypi.tuna.tsinghua.edu.cn/simple' >> \
      '$HOME/.pydistutils.cfg' && mkdir '$HOME/.pip' && \
      echo '[global]' > '$HOME/.pip/pip.conf' && \
      echo 'index_url = https://pypi.tuna.tsinghua.edu.cn/simple' >> '$HOME/.pip/pip.conf'" 
  echo "${config_script}"
}

function main(){
    docker pull $IMG

    docker ps -a --format "{{.Names}}" | grep "${DOCKER_NAME}" 1>/dev/null
    if [ $? == 0 ]; then
        docker stop ${DOCKER_NAME} 1>/dev/null
        docker rm -f ${DOCKER_NAME} 1>/dev/null
    fi
    local display=""
    if [[ -z ${DISPLAY} ]];then
        display=":0"
    else
        display="${DISPLAY}"
    fi

    DOCKER_CMD="nvidia-docker"
    if ! [ -x "$(command -v ${DOCKER_CMD})" ]; then
      DOCKER_CMD="docker"
    fi

    GPU_CONFIG=""
    if [ "$DOCKER_CMD" == "docker" ]; then
      new_docker=$(echo "$(docker -v | cut -c 16-20) >= 19.03" | bc)
      if [ "$new_docker" == "1" ]; then
        GPU_CONFIG="--gpus all"
      fi
    fi

    LOCAL_HOST=`hostname`
    eval ${DOCKER_CMD} run -it \
        -d \
        --name ${DOCKER_NAME}\
        -e DISPLAY=$display \
        -e DOCKER_USER=$USER \
        -e USER=$USER \
        -e DOCKER_USER_ID=$USER_ID \
        -e DOCKER_GRP=$GRP \
        -e DOCKER_GRP_ID=$GRP_ID \
        -e DOCKER_HOME=$DOCKER_HOME \
        -e SSH_AUTH_SOCK=/tmp/.ssh-agent-$USER/agent.sock \
        $(local_volumes) \
        -p :2222 \
        -p :6006 \
        -p :8443 \
        -w /work \
        --dns=114.114.114.114 \
        --add-host in_docker:127.0.0.1 \
        --add-host ${LOCAL_HOST}:127.0.0.1 \
        --hostname in_docker \
        --shm-size 20G \
        $GPU_CONFIG \
        $IMG

    docker exec ${DOCKER_NAME} service ssh start
    if [ "${USER}" != "root" ]; then
        docker exec ${DOCKER_NAME} bash -c "$(add_user)"
    fi

    docker exec -u ${USER} ${DOCKER_NAME} bash -c "$(config_zsh)"
    docker exec -u ${USER} ${DOCKER_NAME} bash -c "$(config_pip)"
    docker exec -d -u $USER ${DOCKER_NAME} /usr/local/code-server/code-server -HN /work

	docker cp -L ~/.gitconfig ${DOCKER_NAME}:${DOCKER_HOME}/.gitconfig
	docker cp -L ~/.vimrc ${DOCKER_NAME}:${DOCKER_HOME}/.vimrc
	docker cp -L ~/.vim ${DOCKER_NAME}:${DOCKER_HOME}
}

main
