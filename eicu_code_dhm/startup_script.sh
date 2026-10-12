#!/bin/bash -l
echo "Startup script is in following directory: $PWD"

# starts the SSH service
/usr/sbin/sshd -D &
service ssh start

poetry shell # this should activate the venv right away even with ssh

# keeps the container alive
sleep infinity

