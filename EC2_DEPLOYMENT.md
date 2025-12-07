# RAG Chatbot - EC2 Deployment Guide

## Prerequisites

1. **AWS EC2 Instance**
   - Instance Type: `t3.medium` or higher (2 vCPU, 4GB RAM minimum)
   - AMI: Ubuntu 22.04 LTS or Amazon Linux 2023
   - Storage: At least 20GB EBS volume
   - Security Group: Allow SSH (port 22) from your IP

2. **Local Requirements**
   - Docker installed locally (for building/testing)
   - SSH key pair for EC2 access
   - AWS CLI configured (optional)

## Deployment Options

### Option 1: Automated Deployment Script (Recommended)

1. **Configure the deployment script:**
   ```bash
   # Edit deploy-ec2.sh and update:
   EC2_HOST="your-ec2-public-dns.compute-1.amazonaws.com"
   EC2_USER="ubuntu"  # or ec2-user for Amazon Linux
   SSH_KEY="~/.ssh/your-key.pem"
   ```

2. **Make script executable:**
   ```bash
   chmod +x deploy-ec2.sh
   ```

3. **Run deployment:**
   ```bash
   ./deploy-ec2.sh
   ```

4. **Update environment variables on EC2:**
   ```bash
   ssh -i ~/.ssh/your-key.pem ubuntu@your-ec2-host
   cd /home/ubuntu/rag-chatbot
   nano .env  # Add your GROQ_API_KEY
   docker-compose restart
   ```

### Option 2: Manual Deployment

#### Step 1: Setup EC2 Instance

SSH into your EC2 instance:
```bash
ssh -i ~/.ssh/your-key.pem ubuntu@your-ec2-host
```

Install Docker:
```bash
# For Ubuntu
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# For Amazon Linux 2023
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user

# Log out and back in for group changes to take effect
```

#### Step 2: Transfer Application Files

From your local machine:
```bash
# Create directory structure
ssh -i ~/.ssh/your-key.pem ubuntu@your-ec2-host "mkdir -p ~/rag-chatbot"

# Copy files using rsync
rsync -avz --progress -e "ssh -i ~/.ssh/your-key.pem" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='venv' \
    ./ ubuntu@your-ec2-host:~/rag-chatbot/

# Or use SCP for individual files
scp -i ~/.ssh/your-key.pem -r ./* ubuntu@your-ec2-host:~/rag-chatbot/
```

#### Step 3: Configure Environment

On EC2:
```bash
cd ~/rag-chatbot

# Create .env file
nano .env
```

Add your configuration:
```env
GROQ_API_KEY=your_actual_api_key_here
LOG_LEVEL=INFO
```

#### Step 4: Build and Run

```bash
# Build Docker image
docker-compose build

# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

## Usage

### Build Index

```bash
docker-compose exec rag-chatbot python main.py build-index \
    --config config.yaml \
    --documents-dir data/documents
```

### Query the Chatbot

```bash
docker-compose exec rag-chatbot python main.py query \
    --config config.yaml
```

### Interactive Shell

```bash
docker-compose exec rag-chatbot bash
```

## Docker Commands

### Container Management

```bash
# Start containers
docker-compose up -d

# Stop containers
docker-compose down

# Restart containers
docker-compose restart

# View logs
docker-compose logs -f

# Check status
docker-compose ps

# Execute commands in container
docker-compose exec rag-chatbot <command>
```

### Image Management

```bash
# Build image
docker-compose build

# Rebuild without cache
docker-compose build --no-cache

# Pull latest images
docker-compose pull

# List images
docker images | grep rag-chatbot

# Remove image
docker rmi rag-chatbot:latest
```

### Cleanup

```bash
# Remove all containers, networks, volumes
docker-compose down -v

# Remove unused images
docker image prune -a

# System cleanup
docker system prune -a --volumes
```

## Monitoring and Troubleshooting

### View Application Logs

```bash
# Container logs
docker-compose logs -f rag-chatbot

# Application logs (mounted volume)
tail -f logs/rag_chatbot.log
```

### Check Resource Usage

```bash
# Container stats
docker stats rag-chatbot

# System resources on EC2
htop  # or top
df -h  # disk usage
free -h  # memory usage
```

### Common Issues

**Issue: Container exits immediately**
```bash
# Check logs
docker-compose logs rag-chatbot

# Check if .env file exists and has correct API key
cat .env
```

**Issue: Out of memory**
```bash
# Adjust resource limits in docker-compose.yml
# Or upgrade to larger EC2 instance type
```

**Issue: Permission denied**
```bash
# Check file ownership
ls -la

# Fix permissions
sudo chown -R $USER:$USER ~/rag-chatbot
```

## Production Best Practices

### 1. Security

- **Never commit .env files** to version control
- Use AWS Secrets Manager or Parameter Store for API keys
- Restrict security group to specific IPs
- Enable SSL/TLS if exposing web interface
- Keep Docker images updated

### 2. Persistence

Mounted volumes ensure data persists:
- `./indexes` - FAISS indexes
- `./logs` - Application logs
- `./data/documents` - Source documents

### 3. Backup Strategy

```bash
# Backup indexes
tar -czf indexes-backup-$(date +%Y%m%d).tar.gz indexes/

# Backup to S3
aws s3 cp indexes-backup-*.tar.gz s3://your-bucket/backups/
```

### 4. Monitoring

Consider adding:
- CloudWatch for EC2 metrics
- Application logging to CloudWatch Logs
- Health check endpoints
- Alert notifications

### 5. Auto-scaling (Advanced)

For production workloads:
- Use ECS/EKS for container orchestration
- Set up Auto Scaling Groups
- Use Application Load Balancer
- Implement health checks

## Cost Optimization

- **Instance Type**: Use t3.medium for development, t3.large+ for production
- **Stop instances** when not in use
- **Use spot instances** for non-critical workloads
- **Monitor usage** with AWS Cost Explorer

## Updating the Application

```bash
# On local machine
git pull  # or make your changes

# Run deployment script again
./deploy-ec2.sh

# Or manually
rsync -avz -e "ssh -i ~/.ssh/your-key.pem" ./ ubuntu@your-ec2-host:~/rag-chatbot/

# On EC2
cd ~/rag-chatbot
docker-compose build
docker-compose up -d
```

## Uninstalling

```bash
# On EC2
cd ~/rag-chatbot
docker-compose down -v
cd ~
rm -rf rag-chatbot

# Remove Docker (optional)
sudo apt-get remove docker.io docker-compose
```

## Support

For issues:
1. Check container logs: `docker-compose logs -f`
2. Check application logs: `cat logs/rag_chatbot.log`
3. Verify .env configuration
4. Ensure EC2 instance has sufficient resources
5. Check security group rules for required ports
