# infra/terraform — AWS eu-central-1 stub

This is a **validated-but-never-applied** Terraform stub. ADR-0010 chose
Hetzner (`docs/DEPLOY.md`, `deploy/deploy.sh`) as the actual hosting plan for
this project — nothing here changes that. This directory exists to
demonstrate infrastructure-as-code literacy and the AWS-side story this
project's Bedrock `eu-central-1` production path (ADR-0002/0004) already
points at, in case AWS is ever the real target instead of Hetzner.

**What it builds (on `apply`, which has never been run):** one EC2
`t3.small` in `eu-central-1` (Frankfurt), an Ubuntu 24.04 AMI resolved via
data source, a security group open on 22/80/443 only, and `user_data` that
fetches and runs the same `deploy/deploy.sh` the Hetzner runbook uses.

**What's out of scope on purpose:** no RDS (Postgres runs in the same
Compose stack `deploy.sh` brings up, same as Hetzner), no ALB/ACM (Caddy
still terminates TLS itself, same reasoning as ADR-0010), no VPC/subnet
resources (uses the account's default VPC — a stub demonstrating the
pattern, not a hardened multi-AZ network). No remote backend/state locking —
state stays local, since this has never been applied and there's no shared
state to protect.

## Validate (no AWS credentials needed)

The `terraform` binary isn't installed locally — use the official Docker
image:

```bash
docker run --rm -v "$PWD":/w -w /w hashicorp/terraform:latest fmt -check
docker run --rm -v "$PWD":/w -w /w hashicorp/terraform:latest init -backend=false
docker run --rm -v "$PWD":/w -w /w hashicorp/terraform:latest validate
```

`make deploy-validate` (repo root) runs all of this plus `bash -n` on
`deploy/deploy.sh`.

## Why not this instead of Hetzner (ADR-0010's reasoning, restated)

A bare Hetzner VPS the project configures end-to-end (Caddy, ufw, Docker
Compose, backups) is a stronger "I control the EU-residency story" narrative
for a portfolio project than a managed AWS EC2 instance — even though AWS
would objectively offer more managed pieces (RDS, ALB, ACM) to lean on.
`hcloud` (Hetzner's own Terraform provider) is the natural next upgrade if
this project ever wants IaC on its actual hosting target, not this AWS stub.
