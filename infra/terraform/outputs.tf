# infra/terraform/outputs.tf

output "public_ip" {
  description = "Public IPv4 of the instance — point DEPLOY_HOSTNAME's DNS A record here."
  value       = aws_instance.app.public_ip
}
