#!/bin/bash

# Define the password as a variable to keep the script clean
PASS='A#TeC#$49357'

echo "Starting Forcepoint VPN in Terminal 1..."
# Opens a new terminal, runs the VPN command, and stays open (exec bash)
gnome-terminal -- bash -c "echo 'y' | sudo forcepoint-client 45.240.51.98 --port 4443 --user architect --password '$PASS' --certaccept; exec bash"

# Optional: Wait a few seconds for the VPN to establish before starting SSH
sleep 5

echo "Starting SSH Tunnel in Terminal 2..."
# Opens a second terminal and uses sshpass to input the password automatically
gnome-terminal -- bash -c "sshpass -p '$PASS' ssh -o StrictHostKeyChecking=no -L 11434:localhost:11434 architect@10.55.205.205; exec bash"

sleep 5

echo "Starting Docker Container in Terminal 3..."

gnome-terminal -- bash -c "sudo docker compose up"
