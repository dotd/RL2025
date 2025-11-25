# DigitalOcean CLI (doctl) Tutorial: Managing Droplets

This tutorial covers the basics of using `doctl` to create, access, list, and delete DigitalOcean droplets.

## Prerequisites

### 1. Install doctl

**macOS:**
```bash
brew install doctl
```

**Linux:**
```bash
cd ~
wget https://github.com/digitalocean/doctl/releases/download/v1.104.0/doctl-1.104.0-linux-amd64.tar.gz
tar xf doctl-1.104.0-linux-amd64.tar.gz
sudo mv doctl /usr/local/bin
```

**Windows:**
Download from the [GitHub releases page](https://github.com/digitalocean/doctl/releases) and add to your PATH.

### 2. Authenticate doctl

First, generate a DigitalOcean API token:
1. Log into your DigitalOcean account
2. Go to API → Tokens/Keys
3. Click "Generate New Token"
4. Give it a name and select read/write permissions
5. Copy the token (you'll only see it once!)

Now authenticate doctl:
```bash
doctl auth init
```

Paste your API token when prompted. You can verify authentication with:
```bash
doctl account get
```

## Step 1: Create SSH Keys (if you don't have them)

Before creating a droplet, you'll want to add your SSH key to DigitalOcean so you can access it.

Generate an SSH key if needed:
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Add your SSH key to DigitalOcean:
```bash
doctl compute ssh-key import my-key --public-key-file ~/.ssh/id_ed25519.pub
```

List your SSH keys to get the key ID or fingerprint:
```bash
doctl compute ssh-key list
```

## Step 2: Create a Droplet

### Explore Available Options

Before creating a droplet, you might want to see what's available:

**List available regions:**
```bash
doctl compute region list
```

**List available droplet sizes:**
```bash
doctl compute size list
```

**List available images:**
```bash
doctl compute image list --public | grep Ubuntu
```

### Create Your Droplet

Here's a basic command to create a droplet:

```bash
doctl compute droplet create my-tutorial-droplet \
  --region nyc3 \
  --size s-1vcpu-1gb \
  --image ubuntu-22-04-x64 \
  --ssh-keys $(doctl compute ssh-key list --format ID --no-header)
```

**Parameters explained:**
- `my-tutorial-droplet` - name of your droplet
- `--region nyc3` - New York datacenter 3
- `--size s-1vcpu-1gb` - 1 CPU, 1GB RAM (basic tier)
- `--image ubuntu-22-04-x64` - Ubuntu 22.04 LTS
- `--ssh-keys` - automatically includes all your SSH keys

The command will output information about the newly created droplet, including its ID and IP address.

**Wait for the droplet to be ready:**
```bash
doctl compute droplet get my-tutorial-droplet
```

Look for the "Status" field to show "active".

## Step 3: List Your Droplets

To see all your running droplets:

```bash
doctl compute droplet list
```

For a more compact view with specific columns:
```bash
doctl compute droplet list --format ID,Name,PublicIPv4,Status,Region
```

To get details about a specific droplet:
```bash
doctl compute droplet get my-tutorial-droplet
```

Or by ID:
```bash
doctl compute droplet get <DROPLET_ID>
```

## Step 4: SSH into Your Droplet

### Get the IP Address

First, get your droplet's IP address:
```bash
doctl compute droplet list --format Name,PublicIPv4 --no-header | grep my-tutorial-droplet
```

Or save it to a variable:
```bash
DROPLET_IP=$(doctl compute droplet list --format Name,PublicIPv4 --no-header | grep my-tutorial-droplet | awk '{print $2}')
echo $DROPLET_IP
```

### Connect via SSH

```bash
ssh root@$DROPLET_IP
```

Or directly:
```bash
ssh root@<YOUR_DROPLET_IP>
```

On first connection, you'll be asked to verify the host fingerprint. Type `yes` to continue.

### Quick Test

Once connected, try running:
```bash
uname -a
hostname
exit
```

## Step 5: Delete Your Droplet

When you're done, clean up by deleting the droplet:

**Delete by name:**
```bash
doctl compute droplet delete my-tutorial-droplet
```

**Delete by ID:**
```bash
doctl compute droplet delete <DROPLET_ID>
```

You'll be prompted to confirm. To skip the confirmation:
```bash
doctl compute droplet delete my-tutorial-droplet --force
```

**Verify deletion:**
```bash
doctl compute droplet list
```

Your tutorial droplet should no longer appear in the list.

## Bonus: Useful Commands

### Create a droplet with a specific tag
```bash
doctl compute droplet create my-droplet \
  --region nyc3 \
  --size s-1vcpu-1gb \
  --image ubuntu-22-04-x64 \
  --ssh-keys $(doctl compute ssh-key list --format ID --no-header) \
  --tag-names tutorial,temporary
```

### List droplets by tag
```bash
doctl compute droplet list --tag-name tutorial
```

### Power off a droplet (without deleting)
```bash
doctl compute droplet-action power-off my-tutorial-droplet
```

### Power on a droplet
```bash
doctl compute droplet-action power-on my-tutorial-droplet
```

### Take a snapshot
```bash
doctl compute droplet-action snapshot my-tutorial-droplet --snapshot-name my-snapshot
```

### Get droplet actions/history
```bash
doctl compute droplet-action list <DROPLET_ID>
```

## Complete Example Workflow

Here's a complete script that creates, tests, and deletes a droplet:

```bash
#!/bin/bash

# Create droplet
echo "Creating droplet..."
doctl compute droplet create test-droplet \
  --region nyc3 \
  --size s-1vcpu-1gb \
  --image ubuntu-22-04-x64 \
  --ssh-keys $(doctl compute ssh-key list --format ID --no-header) \
  --wait

# Get IP address
echo "Getting IP address..."
DROPLET_IP=$(doctl compute droplet list --format Name,PublicIPv4 --no-header | grep test-droplet | awk '{print $2}')
echo "Droplet IP: $DROPLET_IP"

# Wait a moment for SSH to be ready
echo "Waiting for SSH to be ready..."
sleep 30

# SSH and run a command
echo "Testing SSH connection..."
ssh -o StrictHostKeyChecking=no root@$DROPLET_IP 'echo "Hello from droplet!" && hostname'

# List droplets
echo "Current droplets:"
doctl compute droplet list --format Name,PublicIPv4,Status

# Delete droplet
echo "Deleting droplet..."
doctl compute droplet delete test-droplet --force

echo "Done!"
```

## Tips and Best Practices

1. **Always use tags** to organize your droplets, especially in production
2. **Set up firewall rules** using `doctl compute firewall` commands
3. **Use snapshots** for backups before major changes
4. **Monitor costs** by regularly checking unused droplets
5. **Use `--wait` flag** when creating droplets to ensure they're fully ready
6. **Save common configurations** as scripts or shell functions

## Getting Help

View all available commands:
```bash
doctl compute droplet --help
```

For specific command help:
```bash
doctl compute droplet create --help
```

---

**Happy cloud computing with DigitalOcean!** 🚀