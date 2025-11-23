# Reinforcement Learning 2025 Version

## Some Technical stuff

### Command to install locally 

How to install locally the RL2025 package with the file `setup.py`
```
pip install -e .
```
or if there is a ./venv/bin/pip
```
./venv/bin/pip install -e .
```

## Digital Ocean Operating
The commnad is `doctl`. 



```bash
# To init the account we need to do:
doctl auth init --context context01

# List all contexts:
doctl auth list

# Switch to context:
doctl auth switch --context context01

# check whether it works.
# shows the context information
doctl account get 

# Show the ssh keys I have:
ls ~/.ssh/

# Add a specific ssh key and show the ssh keys
doctl compute ssh-key import my-key --public-key-file ~/.ssh/id_rsa.pub
doctl compute ssh-key list
doctl compute ssh-key list --format ID --no-header

# List available regions, available droplet sizes, available images
doctl compute region list
doctl compute size list
doctl compute image list --public

# To confirm that you have successfully granted write access to doctl, create an Ubuntu 24.04 Droplet in the SFO2 region by running:
# doctl compute droplet create --region sfo2 --image ubuntu-24-04-x64 --size s-1vcpu-1gb droplet01
doctl compute droplet create my-tutorial-droplet \
  --region nyc3 \
  --size s-1vcpu-1gb \
  --image ubuntu-22-04-x64 \
  --ssh-keys $(doctl compute ssh-key list --format ID --no-header)

# Show all droplets
doctl compute droplet list

# Get the IP of the droplet
DROPLET_IP=$(doctl compute droplet list --format Name,PublicIPv4 --no-header | grep my-tutorial-droplet | awk '{print $2}')
echo $DROPLET_IP

# Do the ssh:
ssh root@$DROPLET_IP

# Delete droplet:
# doctl compute droplet delete droplet01
doctl compute droplet delete my-tutorial-droplet
```
