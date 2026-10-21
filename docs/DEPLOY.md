# Deploying the Polaris search server on AWS

Every step that creates an AWS resource or costs money needs the team's go-ahead first (CLAUDE.md).
Replace the ALL_CAPS placeholders. Run AWS commands from a laptop with AWS CLI v2 configured
(`brew install awscli`, then `aws configure` or `aws sso login`).

## 0. Facts to collect first

| Item | Where it comes from |
|---|---|
| `REGION` | the organizer's published runner region (network time counts toward p95) |
| `RUNNER_IP` | the organizer's published runner source IP |
| `TEAM_IP` | `curl -s https://checkip.amazonaws.com` on each team laptop |
| `MODEL_ID` | `python scripts/list_models.py --region REGION` (fastest Haiku-class Claude model or its inference profile) |

## 1. Bedrock access (once)

1. Console → Amazon Bedrock → Model access (in `REGION`): enable the Claude Haiku-class model. Anthropic models ask
   for a one-time use-case form.
2. Check quotas (Service Quotas → Amazon Bedrock): on-demand requests per minute and tokens per minute for that
   model. One semantic query makes about 8 LLM calls; the 70 semantic benchmark cases run sequentially, so plan for
   roughly 80 requests/min and 250k input tokens/min. A cross-region inference profile (`us.` / `eu.` / `apac.`
   prefix) raises the limits. If quotas are lower, set `TWIN_POOL=60` (fewer judge calls per query).
3. Probe it: `python scripts/list_models.py --region REGION --probe MODEL_ID` prints the latency of one call.

## 2. IAM role for the instance

```sh
cat > trust.json <<'EOF'
{"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"},
 "Action": "sts:AssumeRole"}]}
EOF
cat > bedrock.json <<'EOF'
{"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
 "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream", "bedrock:Converse",
            "bedrock:ListFoundationModels", "bedrock:ListInferenceProfiles", "bedrock:GetInferenceProfile"],
 "Resource": "*"}]}
EOF
aws iam create-role --role-name polaris-search --assume-role-policy-document file://trust.json
aws iam put-role-policy --role-name polaris-search --policy-name bedrock --policy-document file://bedrock.json
aws iam create-instance-profile --instance-profile-name polaris-search
aws iam add-role-to-instance-profile --instance-profile-name polaris-search --role-name polaris-search
```

## 3. Network

```sh
VPC=$(aws ec2 describe-vpcs --region REGION --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SG=$(aws ec2 create-security-group --region REGION --group-name polaris-search --description polaris --vpc-id $VPC --query GroupId --output text)
aws ec2 authorize-security-group-ingress --region REGION --group-id $SG --protocol tcp --port 8000 --cidr RUNNER_IP/32
aws ec2 authorize-security-group-ingress --region REGION --group-id $SG --protocol tcp --port 8000 --cidr TEAM_IP/32
aws ec2 authorize-security-group-ingress --region REGION --group-id $SG --protocol tcp --port 22 --cidr TEAM_IP/32
```

## 4. Instance (non-burstable; never t-series: CPU credits can throttle mid-run)

```sh
aws ec2 create-key-pair --region REGION --key-name polaris --query KeyMaterial --output text > polaris.pem && chmod 600 polaris.pem
AMI=$(aws ssm get-parameter --region REGION --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 --query Parameter.Value --output text)
aws ec2 run-instances --region REGION --image-id $AMI --instance-type c7i.xlarge --key-name polaris \
  --security-group-ids $SG --iam-instance-profile Name=polaris-search \
  --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=30,VolumeType=gp3}' \
  --metadata-options HttpTokens=required,HttpPutResponseHopLimit=2 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=polaris-search}]'
# Elastic IP so the submitted URL never changes:
aws ec2 allocate-address --region REGION --query AllocationId --output text
aws ec2 associate-address --region REGION --instance-id INSTANCE_ID --allocation-id ALLOCATION_ID
```

`HttpPutResponseHopLimit=2` matters: with the default of 1, boto3 inside a Docker container cannot reach the instance
metadata service, so it gets no Bedrock credentials and the server silently falls back to the rule path.

## 5. Build and run on the instance

```sh
ssh -i polaris.pem ec2-user@PUBLIC_IP
sudo dnf install -y docker git git-lfs && sudo systemctl enable --now docker && sudo usermod -aG docker ec2-user && exit
ssh -i polaris.pem ec2-user@PUBLIC_IP
git lfs install && git clone -b engine https://github.com/VijayabaskarR-06/Polaris-Hackathon-Base-MCP.git polaris
cd polaris && git lfs pull && sha256sum data/products.json   # must be 05b6aa5e...c073
docker build -t polaris-search .
docker run -d --restart unless-stopped --name polaris -p 8000:8000 \
  -e MCP_ALLOWED_HOSTS='localhost:*,127.0.0.1:*,[::1]:*,PUBLIC_IP:*,PUBLIC_IP' \
  -e LLM_PROVIDER=bedrock -e AWS_REGION=REGION \
  -e PLANNER_MODEL_ID=MODEL_ID -e JUDGE_MODEL_ID=MODEL_ID \
  polaris-search
docker logs -f polaris        # wait for "ready: 105126 products"
```

The image runs one uvicorn process (the semantic cache must be shared) and loads everything at startup (~15 s).

## 6. Check from a laptop outside AWS before every submission

```sh
python smoke_test.py http://PUBLIC_IP:8000/mcp
curl -s -o /dev/null -w '%{http_code}\n' -H 'Accept: text/event-stream' http://PUBLIC_IP:8000/mcp   # 405
curl -s http://PUBLIC_IP:8000/healthz
python eval/perf.py http://PUBLIC_IP:8000/mcp
python eval/run_eval.py --url http://PUBLIC_IP:8000/mcp --judge none
```

During a benchmark run: no deploys, restarts or rebuilds (a restart empties the result cache).

## 7. Updating

```sh
cd polaris && git pull && docker build -t polaris-search . && docker rm -f polaris && <the docker run above>
```

## 8. Tear down after the event

Terminate the instance, release the Elastic IP, delete the security group, instance profile and role.
