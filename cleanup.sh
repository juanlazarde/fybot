#!/usr/bin/env bash

# Use when updating fybot's Dockerfile
# Usage: . cleandocker.sh

docker-compose down
docker container rm fybot
docker image prune
docker image rm local/fybot:beta
docker image rm adminer
docker image rm postgres:13
docker volume prune