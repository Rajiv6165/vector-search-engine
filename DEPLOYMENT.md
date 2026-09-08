# Deployment Guide

This guide walks you through deploying the Vector Search Engine to production using Docker. 

## Platform 1: Render (Recommended)

Render makes deploying Dockerized web services incredibly simple.

1. Create an account on [Render](https://render.com/).
2. Click **New +** and select **Web Service**.
3. Connect your GitHub repository.
4. Render will automatically detect the `Dockerfile` in the root of your repository.
5. **Configuration**:
   - **Environment**: Docker
   - **Instance Type**: Select at least the `Starter` tier (or anything with 512MB+ RAM, as the HNSW index requires memory to hold the graph).
   - **Disks**: Add a Disk mounted to `/app/vsearch_data`. This ensures that your vector data and Write-Ahead Log (WAL) persist across deployments and restarts.
6. Click **Create Web Service**. 

Once deployed, your service will have a live URL (e.g., `https://vsearch.onrender.com`). You can pass this URL to your `VSearchClient`!

## Platform 2: Fly.io

[Fly.io](https://fly.io/) is another excellent option for deploying Docker containers close to your users.

1. Install the `flyctl` CLI and authenticate: `fly auth login`.
2. Run `fly launch` in the root of the repository.
   - It will detect the `Dockerfile`.
   - Accept the defaults, but say **Yes** when it asks if you want to tweak settings.
3. In the configuration, create a volume for persistence:
   ```bash
   fly volumes create vsearch_data --region iad --size 1
   ```
4. Update the generated `fly.toml` to mount the volume:
   ```toml
   [mounts]
     source="vsearch_data"
     destination="/app/vsearch_data"
   ```
5. Deploy the application: `fly deploy`.

## Platform 3: Traditional VPS (AWS EC2 / DigitalOcean)

If you are deploying to a traditional Linux server:

1. SSH into your server.
2. Clone the repository.
3. Install Docker and Docker Compose.
4. Run:
   ```bash
   docker-compose up -d --build
   ```
5. We highly recommend putting a reverse proxy like Nginx or Caddy in front of the application to handle SSL/TLS termination.
