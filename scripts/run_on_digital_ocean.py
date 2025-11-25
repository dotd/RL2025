#!/usr/bin/env python3
"""
Script to create a GPU droplet on Digital Ocean.

Usage:
    export DIGITALOCEAN_TOKEN=your_token_here
    python scripts/run_on_digital_ocean.py

Or pass token as argument:
    python scripts/run_on_digital_ocean.py --token your_token_here


"""

import os
import sys
import time
import argparse
import logging
from typing import Optional, Dict, Any

try:
    from pydo import Client
except ImportError:
    print("Error: pydo library is required. Install it with: pip install pydo")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_digitalocean_client(token: Optional[str] = None) -> Client:
    """Initialize Digital Ocean API client."""
    api_token = token or os.environ.get("DIGITALOCEAN_TOKEN")
    if not api_token:
        raise ValueError("DigitalOcean API token not found. " "Set DIGITALOCEAN_TOKEN environment variable or pass --token argument.")
    return Client(token=api_token)


def list_ssh_keys(client: Client) -> list:
    """List available SSH keys."""
    try:
        response = client.ssh_keys.list()
        if response and "ssh_keys" in response:
            return response["ssh_keys"]
        return []
    except Exception as e:
        logger.warning(f"Failed to list SSH keys: {e}")
        return []


def get_ssh_key_ids(client: Client, key_names: Optional[list] = None) -> list:
    """Get SSH key IDs. If key_names is provided, filter by names."""
    ssh_keys = list_ssh_keys(client)
    if not ssh_keys:
        logger.warning("No SSH keys found. Droplet will be created without SSH access.")
        return []

    if key_names:
        # Filter by names
        filtered = [key for key in ssh_keys if key.get("name") in key_names]
        if not filtered:
            logger.warning(f"None of the specified SSH keys found: {key_names}")
            logger.info(f"Available SSH keys: {[k.get('name') for k in ssh_keys]}")
        ssh_keys = filtered

    key_ids = [key["id"] for key in ssh_keys]
    logger.info(f"Using SSH keys: {[k.get('name', 'Unknown') for k in ssh_keys]}")
    return key_ids


def list_available_sizes(client: Client, gpu_only: bool = True) -> list:
    """List available droplet sizes, optionally filtering for GPU sizes."""
    try:
        response = client.sizes.list()
        if not response or "sizes" not in response:
            return []

        sizes = response["sizes"]
        if gpu_only:
            # GPU sizes typically start with 'g-' or have 'gpu' in description
            gpu_sizes = [s for s in sizes if s.get("slug", "").startswith("g-") or "gpu" in s.get("description", "").lower()]
            return gpu_sizes
        return sizes
    except Exception as e:
        logger.warning(f"Failed to list sizes: {e}")
        return []


def list_available_regions(client: Client) -> list:
    """List available regions."""
    try:
        response = client.regions.list()
        if response and "regions" in response:
            return response["regions"]
        return []
    except Exception as e:
        logger.warning(f"Failed to list regions: {e}")
        return []


def wait_for_droplet_ready(client: Client, droplet_id: int, timeout: int = 300) -> bool:
    """Wait for droplet to become active."""
    logger.info(f"Waiting for droplet {droplet_id} to be ready...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = client.droplets.get(droplet_id=droplet_id)
            if response and "droplet" in response:
                droplet = response["droplet"]
                status = droplet.get("status", "unknown")
                logger.info(f"Droplet status: {status}")

                if status == "active":
                    logger.info("Droplet is ready!")
                    return True
                elif status == "new":
                    logger.info("Droplet is being created...")
        except Exception as e:
            logger.warning(f"Error checking droplet status: {e}")

        time.sleep(5)

    logger.error(f"Droplet did not become active within {timeout} seconds")
    return False


def create_gpu_droplet(
    client: Client,
    name: str = "gpu-droplet",
    region: str = "nyc3",
    size: str = "g-2vcpu-8gb",
    image: str = "ubuntu-22-04-x64",
    ssh_key_names: Optional[list] = None,
    ssh_key_ids: Optional[list] = None,
    user_data: Optional[str] = None,
    tags: Optional[list] = None,
    wait: bool = True,
    monitoring: bool = True,
    backups: bool = False,
) -> Optional[Dict[str, Any]]:
    """Create a GPU droplet on Digital Ocean."""
    logger.info(f"Creating GPU droplet '{name}'...")

    # Get SSH key IDs
    if ssh_key_ids is None:
        ssh_key_ids = get_ssh_key_ids(client, ssh_key_names)

    # Prepare droplet configuration
    droplet_config: Dict[str, Any] = {
        "name": name,
        "region": region,
        "size": size,
        "image": image,
        "monitoring": monitoring,
        "backups": backups,
        "ipv6": True,
    }

    if ssh_key_ids:
        droplet_config["ssh_keys"] = ssh_key_ids

    if tags:
        droplet_config["tags"] = tags

    if user_data:
        droplet_config["user_data"] = user_data

    # Create the droplet
    try:
        response = client.droplets.create(body=droplet_config)
        if not response or "droplet" not in response:
            logger.error("Failed to create droplet: Invalid response from API")
            return None

        droplet = response["droplet"]
        droplet_id = droplet["id"]
        logger.info(f"Droplet created successfully! ID: {droplet_id}")

        # Wait for droplet to be ready if requested
        if wait:
            if wait_for_droplet_ready(client, droplet_id):
                # Get updated droplet info with IP address
                updated_response = client.droplets.get(droplet_id=droplet_id)
                if updated_response and "droplet" in updated_response:
                    droplet = updated_response["droplet"]
                    networks = droplet.get("networks", {}).get("v4", [])
                    ipv4 = next((net["ip_address"] for net in networks if net["type"] == "public"), None)
                    if ipv4:
                        logger.info(f"Droplet IP address: {ipv4}")
                        droplet["ip_address"] = ipv4

        return droplet

    except Exception as e:
        logger.error(f"Failed to create droplet: {e}")
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create a GPU droplet on Digital Ocean")
    parser.add_argument("--token", type=str, help="DigitalOcean API token (or set DIGITALOCEAN_TOKEN env var)")
    parser.add_argument("--name", type=str, default="gpu-droplet", help="Droplet name (default: gpu-droplet)")
    parser.add_argument("--region", type=str, default="nyc3", help="Region (default: nyc3)")
    parser.add_argument("--size", type=str, default="g-2vcpu-8gb", help="Droplet size slug (default: g-2vcpu-8gb)")
    parser.add_argument("--image", type=str, default="ubuntu-22-04-x64", help="Image slug (default: ubuntu-22-04-x64)")
    parser.add_argument("--ssh-keys", type=str, nargs="+", help="SSH key names to use")
    parser.add_argument("--tags", type=str, nargs="+", help="Tags for the droplet")
    parser.add_argument("--user-data", type=str, help="Path to user-data script file")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for droplet to be ready")
    parser.add_argument("--list-sizes", action="store_true", help="List available GPU sizes and exit")
    parser.add_argument("--list-regions", action="store_true", help="List available regions and exit")
    parser.add_argument("--list-ssh-keys", action="store_true", help="List available SSH keys and exit")

    args = parser.parse_args()

    try:
        client = get_digitalocean_client(args.token)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # List operations
    if args.list_sizes:
        logger.info("Available GPU sizes:")
        sizes = list_available_sizes(client, gpu_only=True)
        for size in sizes:
            logger.info(
                f"  {size.get('slug')}: {size.get('description')} "
                f"(${size.get('price_monthly', 0)}/month) - {size.get('memory', 0)}MB RAM, "
                f"{size.get('vcpus', 0)} vCPUs"
            )
        return

    if args.list_regions:
        logger.info("Available regions:")
        regions = list_available_regions(client)
        for region in regions:
            available = region.get("available", False)
            status = "✓" if available else "✗"
            logger.info(f"  {status} {region.get('slug')}: {region.get('name')} - {region.get('sizes', [])}")
        return

    if args.list_ssh_keys:
        logger.info("Available SSH keys:")
        ssh_keys = list_ssh_keys(client)
        for key in ssh_keys:
            logger.info(f"  {key.get('id')}: {key.get('name')} ({key.get('fingerprint', 'Unknown')})")
        return

    # Read user-data if provided
    user_data = None
    if args.user_data:
        try:
            with open(args.user_data, "r") as f:
                user_data = f.read()
        except Exception as e:
            logger.error(f"Failed to read user-data file: {e}")
            sys.exit(1)

    # Create droplet
    droplet = create_gpu_droplet(
        client=client,
        name=args.name,
        region=args.region,
        size=args.size,
        image=args.image,
        ssh_key_names=args.ssh_keys,
        user_data=user_data,
        tags=args.tags,
        wait=not args.no_wait,
    )

    if droplet:
        logger.info("Droplet creation completed successfully!")
        logger.info(f"  ID: {droplet.get('id')}")
        logger.info(f"  Name: {droplet.get('name')}")
        logger.info(f"  Status: {droplet.get('status')}")
        if "ip_address" in droplet:
            logger.info(f"  IP Address: {droplet['ip_address']}")
        logger.info(f"  Region: {droplet.get('region', {}).get('slug', 'Unknown')}")
        logger.info(f"  Size: {droplet.get('size_slug', 'Unknown')}")
    else:
        logger.error("Failed to create droplet")
        sys.exit(1)


if __name__ == "__main__":
    main()
