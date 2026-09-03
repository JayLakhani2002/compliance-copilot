# infra/terraform/variables.tf — inputs for the AWS eu-central-1 stub
# (ADR-0010, ADR-0033). No defaults for anything secret/environment-specific
# that would be wrong to bake in; sensible defaults for everything else.

variable "aws_region" {
  description = "AWS region — eu-central-1 (Frankfurt) for EU residency, pairing with the Bedrock eu-central-1 production path (ADR-0002/0004)."
  type        = string
  default     = "eu-central-1"
}

variable "instance_type" {
  description = "EC2 instance type. t3.small mirrors the Hetzner runbook's 4GB-class sizing (ADR-0010's planner amendment)."
  type        = string
  default     = "t3.small"
}

variable "key_name" {
  description = "Name of an EC2 key pair already created in this AWS account/region — used for SSH access instead of a password (same no-password-auth policy as docs/DEPLOY.md)."
  type        = string
}

variable "deploy_hostname" {
  description = "Public hostname this instance will serve (Caddyfile's DEPLOY_HOSTNAME) — DNS must point here separately, same as docs/DEPLOY.md step 3."
  type        = string
}

variable "repo_raw_url" {
  description = "Raw base URL to fetch deploy/deploy.sh from at boot, e.g. https://raw.githubusercontent.com/<you>/compliance-copilot/main."
  type        = string
}

variable "repo_url" {
  description = "Git clone URL deploy.sh uses to clone the app repo onto the instance."
  type        = string
}

variable "app_user" {
  description = "Non-root sudo user deploy.sh creates and runs the app under."
  type        = string
  default     = "deploy"
}
