"""
Docker Deployment Manager

Comprehensive Docker containerization and deployment management for the OpenSports platform
with multi-stage builds, optimization, security scanning, and container orchestration.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

import asyncio
import docker
import os
import yaml
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import tempfile
from opensports.core.config import settings
from opensports.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DockerConfig:
    """Docker deployment configuration."""
    image_name: str
    tag: str = "latest"
    registry: Optional[str] = None
    dockerfile_path: str = "Dockerfile"
    build_context: str = "."
    build_args: Dict[str, str] = field(default_factory=dict)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    ports: List[Dict[str, int]] = field(default_factory=list)
    volumes: List[Dict[str, str]] = field(default_factory=list)
    networks: List[str] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    health_check: Optional[Dict[str, Any]] = None
    resource_limits: Dict[str, str] = field(default_factory=dict)


class DockerManager:
    """
    Comprehensive Docker deployment manager.
    
    Features:
    - Multi-stage Docker builds
    - Image optimization and security scanning
    - Container orchestration
    - Health monitoring
    - Automated deployment pipelines
    - Registry management
    - Rollback capabilities
    """
    
    def __init__(self):
        self.client = None
        self.registry_auth = {}
        self.deployment_history = []
        
    async def initialize(self):
        """Initialize Docker client and configuration."""
        try:
            self.client = docker.from_env()
            logger.info("Docker client initialized successfully")
            
            # Test Docker connection
            version = self.client.version()
            logger.info(f"Docker version: {version['Version']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise
    
    def generate_dockerfile(self, config: DockerConfig) -> str:
        """Generate optimized Dockerfile for OpenSports platform."""
        dockerfile_content = f"""
# OpenSports Platform Dockerfile
# Multi-stage build for optimized production deployment
# Author: Nik Jois (nikjois@llamaearch.ai)

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_ENV=production
ARG VERSION=latest

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the application
RUN pip install --no-cache-dir -e .

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/app/.venv/bin:$PATH"

# Create non-root user
RUN groupadd -r opensports && useradd -r -g opensports opensports

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models && \\
    chown -R opensports:opensports /app

# Switch to non-root user
USER opensports

# Expose ports
EXPOSE 8000 8001 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "opensports.api.main"]
"""
        
        # Add custom build args
        if config.build_args:
            build_args_section = "\n".join([f"ARG {k}={v}" for k, v in config.build_args.items()])
            dockerfile_content = dockerfile_content.replace("ARG VERSION=latest", f"ARG VERSION=latest\n{build_args_section}")
        
        return dockerfile_content
    
    def generate_docker_compose(self, config: DockerConfig) -> str:
        """Generate Docker Compose configuration."""
        compose_config = {
            'version': '3.8',
            'services': {
                'opensports-api': {
                    'build': {
                        'context': config.build_context,
                        'dockerfile': config.dockerfile_path,
                        'args': config.build_args
                    },
                    'image': f"{config.image_name}:{config.tag}",
                    'container_name': 'opensports-api',
                    'restart': 'unless-stopped',
                    'ports': config.ports or [{'8000': 8000}],
                    'environment': config.environment_vars,
                    'volumes': config.volumes or [
                        './data:/app/data',
                        './logs:/app/logs',
                        './models:/app/models'
                    ],
                    'networks': config.networks or ['opensports-network'],
                    'labels': config.labels,
                    'healthcheck': config.health_check or {
                        'test': ['CMD', 'curl', '-f', 'http://localhost:8000/health'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3,
                        'start_period': '40s'
                    }
                },
                'opensports-worker': {
                    'build': {
                        'context': config.build_context,
                        'dockerfile': config.dockerfile_path,
                        'args': config.build_args
                    },
                    'image': f"{config.image_name}:{config.tag}",
                    'container_name': 'opensports-worker',
                    'restart': 'unless-stopped',
                    'command': ['python', '-m', 'opensports.realtime.stream_processor'],
                    'environment': config.environment_vars,
                    'volumes': config.volumes or [
                        './data:/app/data',
                        './logs:/app/logs',
                        './models:/app/models'
                    ],
                    'networks': config.networks or ['opensports-network'],
                    'depends_on': ['redis', 'opensports-api']
                },
                'redis': {
                    'image': 'redis:7-alpine',
                    'container_name': 'opensports-redis',
                    'restart': 'unless-stopped',
                    'ports': [{'6379': 6379}],
                    'volumes': ['redis-data:/data'],
                    'networks': ['opensports-network'],
                    'command': 'redis-server --appendonly yes'
                },
                'nginx': {
                    'image': 'nginx:alpine',
                    'container_name': 'opensports-nginx',
                    'restart': 'unless-stopped',
                    'ports': [{'80': 80}, {'443': 443}],
                    'volumes': [
                        './nginx.conf:/etc/nginx/nginx.conf:ro',
                        './ssl:/etc/nginx/ssl:ro'
                    ],
                    'networks': ['opensports-network'],
                    'depends_on': ['opensports-api']
                },
                'monitoring': {
                    'build': {
                        'context': config.build_context,
                        'dockerfile': config.dockerfile_path,
                        'args': config.build_args
                    },
                    'image': f"{config.image_name}:{config.tag}",
                    'container_name': 'opensports-monitoring',
                    'restart': 'unless-stopped',
                    'ports': [{'8501': 8501}],
                    'command': ['streamlit', 'run', 'opensports/monitoring/dashboard.py'],
                    'environment': config.environment_vars,
                    'volumes': config.volumes or [
                        './data:/app/data',
                        './logs:/app/logs'
                    ],
                    'networks': ['opensports-network'],
                    'depends_on': ['opensports-api', 'redis']
                }
            },
            'networks': {
                'opensports-network': {
                    'driver': 'bridge'
                }
            },
            'volumes': {
                'redis-data': {},
                'opensports-data': {},
                'opensports-logs': {},
                'opensports-models': {}
            }
        }
        
        return yaml.dump(compose_config, default_flow_style=False)
    
    def generate_nginx_config(self) -> str:
        """Generate Nginx configuration for load balancing and SSL termination."""
        nginx_config = """
events {
    worker_connections 1024;
}

http {
    upstream opensports_api {
        server opensports-api:8000;
    }
    
    upstream opensports_monitoring {
        server opensports-monitoring:8501;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=monitoring:10m rate=5r/s;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;
    
    # API server
    server {
        listen 80;
        server_name api.opensports.local;
        
        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name api.opensports.local;
        
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        
        location / {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://opensports_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }
        
        location /health {
            access_log off;
            proxy_pass http://opensports_api/health;
        }
    }
    
    # Monitoring dashboard
    server {
        listen 80;
        server_name monitoring.opensports.local;
        
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name monitoring.opensports.local;
        
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        
        location / {
            limit_req zone=monitoring burst=10 nodelay;
            
            proxy_pass http://opensports_monitoring;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support for Streamlit
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
"""
        return nginx_config
    
    async def build_image(self, config: DockerConfig, push: bool = False) -> str:
        """Build Docker image with optimization."""
        logger.info(f"Building Docker image: {config.image_name}:{config.tag}")
        
        try:
            # Generate Dockerfile if it doesn't exist
            dockerfile_path = Path(config.dockerfile_path)
            if not dockerfile_path.exists():
                dockerfile_content = self.generate_dockerfile(config)
                dockerfile_path.write_text(dockerfile_content)
                logger.info(f"Generated Dockerfile at {dockerfile_path}")
            
            # Build image
            image, build_logs = self.client.images.build(
                path=config.build_context,
                dockerfile=config.dockerfile_path,
                tag=f"{config.image_name}:{config.tag}",
                buildargs=config.build_args,
                rm=True,
                forcerm=True,
                pull=True
            )
            
            # Log build output
            for log in build_logs:
                if 'stream' in log:
                    logger.debug(log['stream'].strip())
            
            logger.info(f"Successfully built image: {image.id}")
            
            # Push to registry if requested
            if push and config.registry:
                await self.push_image(config)
            
            return image.id
            
        except Exception as e:
            logger.error(f"Failed to build Docker image: {e}")
            raise
    
    async def push_image(self, config: DockerConfig) -> bool:
        """Push image to registry."""
        if not config.registry:
            logger.warning("No registry specified, skipping push")
            return False
        
        full_image_name = f"{config.registry}/{config.image_name}:{config.tag}"
        
        try:
            logger.info(f"Pushing image to registry: {full_image_name}")
            
            # Tag image for registry
            image = self.client.images.get(f"{config.image_name}:{config.tag}")
            image.tag(config.registry, config.image_name, config.tag)
            
            # Push image
            push_logs = self.client.images.push(
                repository=f"{config.registry}/{config.image_name}",
                tag=config.tag,
                auth_config=self.registry_auth.get(config.registry)
            )
            
            logger.info(f"Successfully pushed image: {full_image_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to push image: {e}")
            return False
    
    async def deploy_container(self, config: DockerConfig) -> str:
        """Deploy container with health checks."""
        container_name = f"{config.image_name}-{config.tag}"
        
        try:
            logger.info(f"Deploying container: {container_name}")
            
            # Stop existing container if running
            try:
                existing_container = self.client.containers.get(container_name)
                existing_container.stop()
                existing_container.remove()
                logger.info(f"Stopped existing container: {container_name}")
            except docker.errors.NotFound:
                pass
            
            # Prepare container configuration
            container_config = {
                'image': f"{config.image_name}:{config.tag}",
                'name': container_name,
                'environment': config.environment_vars,
                'ports': {str(p['container']): p['host'] for p in config.ports} if config.ports else {},
                'volumes': {v['host']: {'bind': v['container'], 'mode': v.get('mode', 'rw')} for v in config.volumes} if config.volumes else {},
                'labels': config.labels,
                'restart_policy': {'Name': 'unless-stopped'},
                'detach': True
            }
            
            # Add resource limits
            if config.resource_limits:
                container_config['mem_limit'] = config.resource_limits.get('memory', '1g')
                container_config['cpu_quota'] = int(config.resource_limits.get('cpu_quota', 100000))
            
            # Create and start container
            container = self.client.containers.run(**container_config)
            
            # Wait for container to be healthy
            await self._wait_for_health(container, timeout=120)
            
            logger.info(f"Successfully deployed container: {container.id}")
            
            # Record deployment
            self.deployment_history.append({
                'timestamp': datetime.now(),
                'container_id': container.id,
                'image': f"{config.image_name}:{config.tag}",
                'status': 'deployed'
            })
            
            return container.id
            
        except Exception as e:
            logger.error(f"Failed to deploy container: {e}")
            raise
    
    async def deploy_compose(self, config: DockerConfig, compose_file: str = None) -> bool:
        """Deploy using Docker Compose."""
        try:
            if not compose_file:
                # Generate compose file
                compose_content = self.generate_docker_compose(config)
                compose_file = "docker-compose.yml"
                with open(compose_file, 'w') as f:
                    f.write(compose_content)
                logger.info(f"Generated Docker Compose file: {compose_file}")
            
            # Generate Nginx config
            nginx_config = self.generate_nginx_config()
            with open("nginx.conf", 'w') as f:
                f.write(nginx_config)
            
            logger.info("Deploying with Docker Compose")
            
            # Run docker-compose up
            result = subprocess.run([
                'docker-compose', '-f', compose_file, 'up', '-d', '--build'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Docker Compose deployment successful")
                return True
            else:
                logger.error(f"Docker Compose deployment failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to deploy with Docker Compose: {e}")
            return False
    
    async def _wait_for_health(self, container, timeout: int = 120):
        """Wait for container to become healthy."""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < timeout:
            container.reload()
            
            if container.status == 'running':
                # Check health status
                health = container.attrs.get('State', {}).get('Health', {})
                if health.get('Status') == 'healthy':
                    logger.info(f"Container {container.id} is healthy")
                    return
                elif health.get('Status') == 'unhealthy':
                    raise Exception(f"Container {container.id} is unhealthy")
            elif container.status == 'exited':
                logs = container.logs().decode('utf-8')
                raise Exception(f"Container {container.id} exited: {logs}")
            
            await asyncio.sleep(5)
        
        raise Exception(f"Container {container.id} health check timeout")
    
    async def scan_image_security(self, config: DockerConfig) -> Dict[str, Any]:
        """Scan image for security vulnerabilities."""
        image_name = f"{config.image_name}:{config.tag}"
        
        try:
            logger.info(f"Scanning image for vulnerabilities: {image_name}")
            
            # Use Trivy for vulnerability scanning
            result = subprocess.run([
                'trivy', 'image', '--format', 'json', image_name
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                scan_results = json.loads(result.stdout)
                
                # Summarize results
                summary = {
                    'image': image_name,
                    'scan_time': datetime.now().isoformat(),
                    'vulnerabilities': {
                        'critical': 0,
                        'high': 0,
                        'medium': 0,
                        'low': 0
                    },
                    'details': scan_results
                }
                
                # Count vulnerabilities by severity
                for result in scan_results.get('Results', []):
                    for vuln in result.get('Vulnerabilities', []):
                        severity = vuln.get('Severity', '').lower()
                        if severity in summary['vulnerabilities']:
                            summary['vulnerabilities'][severity] += 1
                
                logger.info(f"Security scan completed: {summary['vulnerabilities']}")
                return summary
            else:
                logger.warning(f"Security scan failed: {result.stderr}")
                return {'error': result.stderr}
                
        except FileNotFoundError:
            logger.warning("Trivy not found, skipping security scan")
            return {'error': 'Trivy scanner not available'}
        except Exception as e:
            logger.error(f"Security scan error: {e}")
            return {'error': str(e)}
    
    async def get_container_stats(self, container_id: str) -> Dict[str, Any]:
        """Get container resource usage statistics."""
        try:
            container = self.client.containers.get(container_id)
            stats = container.stats(stream=False)
            
            # Calculate CPU usage percentage
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
            cpu_percent = (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100.0
            
            # Calculate memory usage
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            memory_percent = (memory_usage / memory_limit) * 100.0
            
            return {
                'container_id': container_id,
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_usage_mb': memory_usage / (1024 * 1024),
                'memory_limit_mb': memory_limit / (1024 * 1024),
                'memory_percent': memory_percent,
                'network_rx_bytes': stats['networks']['eth0']['rx_bytes'],
                'network_tx_bytes': stats['networks']['eth0']['tx_bytes'],
                'block_read_bytes': stats['blkio_stats']['io_service_bytes_recursive'][0]['value'],
                'block_write_bytes': stats['blkio_stats']['io_service_bytes_recursive'][1]['value']
            }
            
        except Exception as e:
            logger.error(f"Failed to get container stats: {e}")
            return {}
    
    async def cleanup_old_images(self, keep_count: int = 5):
        """Clean up old Docker images."""
        try:
            images = self.client.images.list()
            
            # Group images by repository
            image_groups = {}
            for image in images:
                for tag in image.tags:
                    if ':' in tag:
                        repo, version = tag.rsplit(':', 1)
                        if repo not in image_groups:
                            image_groups[repo] = []
                        image_groups[repo].append((image, version, tag))
            
            # Remove old images
            removed_count = 0
            for repo, image_list in image_groups.items():
                # Sort by creation date (newest first)
                image_list.sort(key=lambda x: x[0].attrs['Created'], reverse=True)
                
                # Keep only the specified number of images
                for image, version, tag in image_list[keep_count:]:
                    try:
                        self.client.images.remove(tag, force=True)
                        removed_count += 1
                        logger.info(f"Removed old image: {tag}")
                    except Exception as e:
                        logger.warning(f"Failed to remove image {tag}: {e}")
            
            logger.info(f"Cleaned up {removed_count} old images")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old images: {e}")
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        try:
            containers = self.client.containers.list(all=True)
            
            status = {
                'timestamp': datetime.now().isoformat(),
                'total_containers': len(containers),
                'running_containers': len([c for c in containers if c.status == 'running']),
                'containers': []
            }
            
            for container in containers:
                container_info = {
                    'id': container.id[:12],
                    'name': container.name,
                    'image': container.image.tags[0] if container.image.tags else 'unknown',
                    'status': container.status,
                    'created': container.attrs['Created'],
                    'ports': container.ports
                }
                status['containers'].append(container_info)
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            return {'error': str(e)} 