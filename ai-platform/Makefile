.PHONY: help install clean test format lint type-check migrate seed db-shell services-up services-down infrastructure-up infrastructure-down deploy monitoring-up monitoring-down

# Default target
help:
	@echo "GuidedPATH - AI-Powered Healthcare Platform"
	@echo ""
	@echo "Available commands:"
	@echo "  install              Install Python dependencies"
	@echo "  clean               Remove Python cache files"
	@echo "  test                Run test suite"
	@echo "  format              Format code with black and isort"
	@echo "  lint                Lint code with flake8"
	@echo "  type-check          Run mypy type checking"
	@echo "  migrate             Run database migrations"
	@echo "  seed                Seed database with initial data"
	@echo "  db-shell            Open database shell"
	@echo "  services-up         Start all services"
	@echo "  services-down       Stop all services"
	@echo "  infrastructure-up   Set up infrastructure (K8s, DB, etc.)"
	@echo "  infrastructure-down Tear down infrastructure"
	@echo "  deploy              Deploy to production"
	@echo "  monitoring-up       Start monitoring stack"
	@echo "  monitoring-down     Stop monitoring stack"

# Environment setup
install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Code quality
format:
	black .
	isort .

lint:
	flake8 .

type-check:
	mypy .

# Database operations
migrate:
	cd backend && alembic upgrade head

seed:
	cd backend && python -m scripts.seed_database

db-shell:
	docker exec -it guidedpath-postgres psql -U guidedpath -d guidedpath

# Services management
services-up:
	docker-compose -f infrastructure/docker-compose.yml up -d

services-down:
	docker-compose -f infrastructure/docker-compose.yml down

# Infrastructure
infrastructure-up:
	cd infrastructure/terraform && terraform init && terraform apply -auto-approve

infrastructure-down:
	cd infrastructure/terraform && terraform destroy -auto-approve

# Kubernetes deployment
deploy:
	kubectl apply -f infrastructure/kubernetes/
	kubectl rollout status deployment/guidedpath-backend
	kubectl rollout status deployment/guidedpath-frontend

# Monitoring
monitoring-up:
	kubectl apply -f infrastructure/kubernetes/monitoring/
	kubectl port-forward svc/prometheus 9090:9090 &
	kubectl port-forward svc/grafana 3000:3000 &
	kubectl port-forward svc/kibana 5601:5601 &

monitoring-down:
	kubectl delete -f infrastructure/kubernetes/monitoring/

# Testing
test:
	pytest backend/tests/ -v --cov=backend/apps --cov-report=html

# Development
dev-up:
	docker-compose -f infrastructure/docker-compose.dev.yml up

dev-down:
	docker-compose -f infrastructure/docker-compose.dev.yml down

# Production build
build:
	docker build -t guidedpath-backend:latest -f infrastructure/docker/Dockerfile.backend .
	docker build -t guidedpath-frontend:latest -f infrastructure/docker/Dockerfile.frontend .

# Logs
logs:
	kubectl logs -f deployment/guidedpath-backend
	kubectl logs -f deployment/guidedpath-frontend

# Backup
backup:
	./scripts/backup.sh

# Security scan
security-scan:
	safety check
	bandit -r backend/
