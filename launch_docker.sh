#!/bin/bash

# to rebuild the Docker image and launch docker
docker build -t graphsage .
docker run -it graphsage bash
