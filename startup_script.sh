#!/bin/bash
# Note: Permission denied errors related to this file might be related to the CR symbol,
# i.e. remove the empty line at the end of the file if there's any!
echo "Startup script is in following directory: $PWD"
echo "Current user: $(whoami)"
# starts the SSH service

#ssh-keygen -A # fixes: missing host keys: exiting
sudo /usr/sbin/sshd -D &
sudo service ssh start

# keeps the container alive
sleep infinity