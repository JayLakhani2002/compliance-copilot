# infra/terraform/main.tf — AWS eu-central-1 stub (ADR-0010, ADR-0033).
# Validated (fmt/init/validate) but NEVER applied — see README.md in this
# directory for what this demonstrates and why it isn't the chosen path.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  # No backend block — state stays local (this stub is never applied, so a
  # remote backend/locking would be machinery with nothing to protect).
}

provider "aws" {
  region = var.aws_region
}

# Canonical's official Ubuntu 24.04 LTS (Noble) AMI, resolved at plan time
# instead of a hardcoded, region-specific, rot-prone AMI id.
data "aws_ami" "ubuntu_2404" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# 22/80/443 only — the same perimeter ADR-0010's Hetzner ufw rules enforce.
resource "aws_security_group" "app" {
  name        = "compliance-copilot-app"
  description = "22 (SSH), 80/443 (Caddy) inbound; all outbound"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    description = "HTTP (Caddy ACME + redirect)"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# user_data fetches and runs THE SAME deploy/deploy.sh the Hetzner runbook
# uses (docs/DEPLOY.md) — one hardening+docker+clone+compose script, not a
# second copy maintained here. It stops itself before `compose up` if the
# operator hasn't placed a real .env on the instance yet (deploy.sh's own
# check), so first boot never runs an app with placeholder secrets.
locals {
  user_data = <<-EOF
    #!/bin/bash
    set -euo pipefail
    curl -fsSL "${var.repo_raw_url}/deploy/deploy.sh" -o /root/deploy.sh
    chmod +x /root/deploy.sh
    HOSTNAME="${var.deploy_hostname}" REPO_URL="${var.repo_url}" APP_USER="${var.app_user}" bash /root/deploy.sh
  EOF
}

resource "aws_instance" "app" {
  ami                    = data.aws_ami.ubuntu_2404.id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.app.id]
  user_data              = local.user_data

  tags = {
    Name = "compliance-copilot"
  }
}
