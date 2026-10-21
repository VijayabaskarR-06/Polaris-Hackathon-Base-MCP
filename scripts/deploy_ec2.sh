#!/usr/bin/env bash
# One-command deploy of the search server to a fresh EC2 instance.
#   REGION=ap-south-1 [RUNNER_IP=1.2.3.4] scripts/deploy_ec2.sh
# Needs AWS CLI credentials (aws configure). Creates: a security group, a c7i.xlarge
# instance that builds and runs the Docker image from GitHub, and an Elastic IP.
set -euo pipefail
REGION=${REGION:?set REGION to the organizer runner region}
NAME=${NAME:-polaris-search}
TYPE=${TYPE:-c7i.xlarge}
REPO=${REPO:-https://github.com/VijayabaskarR-06/Polaris-Hackathon-Base-MCP.git}
BRANCH=${BRANCH:-main}
MY_IP=$(curl -s https://checkip.amazonaws.com)
SOURCE=${RUNNER_IP:+$RUNNER_IP/32}
SOURCE=${SOURCE:-0.0.0.0/0}

VPC=$(aws ec2 describe-vpcs --region "$REGION" --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SG=$(aws ec2 describe-security-groups --region "$REGION" --filters Name=group-name,Values="$NAME" Name=vpc-id,Values="$VPC" \
     --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
if [ -z "$SG" ] || [ "$SG" = "None" ]; then
  SG=$(aws ec2 create-security-group --region "$REGION" --group-name "$NAME" --description "$NAME" --vpc-id "$VPC" \
       --query GroupId --output text)
  aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG" --protocol tcp --port 8000 --cidr "$SOURCE"
  aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG" --protocol tcp --port 8000 --cidr "$MY_IP/32" 2>/dev/null || true
fi

RAW=${REPO%.git}
RAW=${RAW/github.com/media.githubusercontent.com\/media}
cat > /tmp/polaris-user-data.sh <<EOF
#!/bin/bash
set -x
dnf install -y docker git
dnf install -y git-lfs || true
systemctl enable --now docker
git lfs install --system || true
GIT_LFS_SKIP_SMUDGE=1 git clone -b $BRANCH $REPO /opt/polaris
cd /opt/polaris && (git lfs pull || true)
# Without git-lfs the catalog is a pointer file: fetch the real one and check its SHA-256.
if [ "\$(stat -c %s data/products.json)" -lt 1000000 ]; then
  curl -sSL -o data/products.json "$RAW/$BRANCH/data/products.json"
fi
echo "05b6aa5e8e8c1c8c997d584f8288e432862454f8820531e13473443e83c8f073  data/products.json" | sha256sum -c
docker build -t polaris /opt/polaris
docker run -d --restart unless-stopped --name polaris -p 8000:8000 polaris
EOF

AMI=$(aws ssm get-parameter --region "$REGION" --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
      --query Parameter.Value --output text)
INSTANCE=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type "$TYPE" \
  --security-group-ids "$SG" --user-data file:///tmp/polaris-user-data.sh \
  --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=30,VolumeType=gp3}' \
  --metadata-options HttpTokens=required,HttpPutResponseHopLimit=2 \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" \
  --query 'Instances[0].InstanceId' --output text)
echo "instance $INSTANCE starting..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE"
ALLOC=$(aws ec2 allocate-address --region "$REGION" --query AllocationId --output text)
aws ec2 associate-address --region "$REGION" --instance-id "$INSTANCE" --allocation-id "$ALLOC" >/dev/null
IP=$(aws ec2 describe-addresses --region "$REGION" --allocation-ids "$ALLOC" --query 'Addresses[0].PublicIp' --output text)
echo "public IP $IP; waiting for the server (Docker build takes a few minutes)..."
for i in $(seq 1 120); do
  if curl -s -m 3 "http://$IP:8000/healthz" | grep -q ok; then
    echo "READY: submit http://$IP:8000/mcp"
    exit 0
  fi
  sleep 10
done
echo "server not up yet; check later: curl http://$IP:8000/healthz"
