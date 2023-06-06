#!/bin/bash

function usage {
    echo "Usage: $0 [proxy] [tag]"
    echo "Builds a Docker container with the specified proxy and an optional tag."
    echo "Parameters:"
    echo "  proxy (optional): The HTTP proxy to use during the build."
    echo "  tag (optional): The tag to apply to the Docker container. Defaults to 'pytorch_speech_dev'."
    echo "  -h/--help:   Show this help message."
    exit 1
}

# Check if help parameter is provided
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]
then
    usage
fi

# Assign parameters to variables
proxy=${1:"http://172.29.96.1:7890"}
tag=${2:-"transformer-playground"}

# Build Docker container
cd .. && docker build --build-arg HTTP_PROXY=$proxy -t $tag .

# Check if build was successful
if [ $? -eq 0 ]
then
    echo "Docker container was built successfully."
else
    echo "Failed to build Docker container."
fi
