#!/usr/bin/env bash
# Install Docker and Docker Compose in linux-based (Ubuntu, Debian)
set -o errexit

# Install Docker
printf "\nInstalling Docker\n"
sudo apt update && sudo apt dist-upgrade -y
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update
apt-cache policy docker-ce
sudo apt install -y docker-ce

# Install Docker Compose
printf "\nInstalling Docker Compose\n"
mkdir -p ~/.docker/cli-plugins/
curl -SL https://github.com/docker/compose/releases/download/v2.2.3/docker-compose-linux-x86_64 -o ~/.docker/cli-plugins/docker-compose
chmod +x ~/.docker/cli-plugins/docker-compose
sudo chown "${USER}" /var/run/docker.sock

# Docker Command Without Sudo
printf "\nElevating user privilege to run docker without sudo\n"
sudo usermod -aG docker "${USER}" && su - "${USER}"

# Verify
printf "\nVerifying\n"
sudo systemctl status docker | grep "Active:"
docker info
docker compose version

printf '%s\n' \
"
--- Final thoughts ---

docker stop \$(docker ps -aq) && docker rm \$(docker ps -aq)
nano docker-composer.yaml
docker compose up -d
"