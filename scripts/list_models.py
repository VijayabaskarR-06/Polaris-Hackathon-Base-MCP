"""List the Claude models and inference profiles Amazon Bedrock offers in a region.

    python scripts/list_models.py [--region us-east-1] [--access] [--all-profiles] [--json]
    python scripts/list_models.py --probe MODEL_ID [--probe MODEL_ID ...] [--repeat 3]

A model without ON_DEMAND support is only callable through an inference profile:
use the profile id (for example "us.anthropic...") as PLANNER_MODEL_ID /
JUDGE_MODEL_ID. --access asks Bedrock whether this account may invoke each
model. --probe sends tiny converse() calls through engine.llm.BedrockLLM,
exactly as the server does, and prints their latency; the first call includes
connection setup, and a 400 for an optional field (temperature) shows as a
warning followed by the resent request.

Exit status: 0 ok, 1 an AWS call or probe failed, 2 no boto3, region or credentials.
"""
import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.llm import BedrockLLM  # noqa: E402

PROBE_SYSTEM = 'You are a health check. Reply with the single word OK.'
AUTH_ERRORS = {'UnrecognizedClientException', 'InvalidSignatureException', 'ExpiredToken',
               'ExpiredTokenException', 'InvalidClientTokenId', 'AccessDeniedException', 'AccessDenied'}
NO_CREDENTIALS = ('no AWS credentials found. Run "aws configure" or "aws sso login", set '
                  'AWS_PROFILE or AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, or run on an '
                  'instance with an IAM role.')
HINT = ('Use a model id with on-demand: yes directly; otherwise use an inference profile id.\n'
        'Set it as PLANNER_MODEL_ID / JUDGE_MODEL_ID, then measure it:\n'
        '  python scripts/list_models.py --probe ID')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='List Claude models and inference profiles on Amazon Bedrock.')
    parser.add_argument('--region', default=os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION'),
                        help='AWS region (default: AWS_REGION, AWS_DEFAULT_REGION, the AWS config)')
    parser.add_argument('--access', action='store_true',
                        help='check whether this account may invoke each model')
    parser.add_argument('--all-profiles', action='store_true',
                        help='list every inference profile, not only Claude ones')
    parser.add_argument('--json', action='store_true', help='print the raw API summaries as JSON')
    parser.add_argument('--probe', action='append', metavar='MODEL_ID',
                        help='send tiny converse() calls to this model or profile id (repeatable)')
    parser.add_argument('--repeat', type=int, default=3, help='probe calls per model (default 3)')
    parser.add_argument('--max-tokens', type=int, default=16, help='probe max_tokens (default 16)')
    parser.add_argument('--timeout', type=float, default=15.0,
                        help='seconds per probe call (default 15)')
    return parser.parse_args(argv)


def fail(message, status):
    print(f'error: {message}', file=sys.stderr)
    return status


def client_error(exc):
    """Report a botocore ClientError; credential and permission problems exit with 2."""
    error = exc.response.get('Error') or {}
    code = error.get('Code', 'ClientError')
    message = f"{code}: {error.get('Message', '')}"
    if code in AUTH_ERRORS:
        return fail(message + '\nThe AWS credentials are invalid or expired, or lack '
                    'bedrock:ListFoundationModels and bedrock:ListInferenceProfiles.', 2)
    return fail(message, 1)


def inference_profiles(bedrock):
    profiles = []
    for kind in ('SYSTEM_DEFINED', 'APPLICATION'):
        for page in bedrock.get_paginator('list_inference_profiles').paginate(typeEquals=kind):
            profiles.extend(page.get('inferenceProfileSummaries', []))
    return profiles


def is_claude(profile):
    text = ' '.join([profile.get('inferenceProfileId', ''), profile.get('inferenceProfileName', '')]
                    + [m.get('modelArn', '') for m in profile.get('models') or []]).lower()
    return 'anthropic' in text or 'claude' in text


def model_access(bedrock, model_id):
    """This account's access to a model, in a few words."""
    try:
        r = bedrock.get_foundation_model_availability(modelId=model_id)
    except Exception as exc:  # missing bedrock:GetFoundationModelAvailability, or an old botocore
        code = (getattr(exc, 'response', None) or {}).get('Error', {}).get('Code')
        return f'unknown ({code or type(exc).__name__})'
    agreement = (r.get('agreementAvailability') or {}).get('status', '?')
    return (f"{r.get('authorizationStatus', '?')}, agreement {agreement}, "
            f"entitlement {r.get('entitlementAvailability', '?')}, "
            f"region {r.get('regionAvailability', '?')}")


def arn_region(arn):
    parts = arn.split(':')
    return parts[3] if len(parts) > 5 else ''


def print_models(region, models, access):
    print(f'Claude foundation models in {region}: {len(models)}')
    for m in sorted(models, key=lambda m: m.get('modelId', '')):
        types = m.get('inferenceTypesSupported') or []
        status = (m.get('modelLifecycle') or {}).get('status', '?')
        on_demand = 'yes' if 'ON_DEMAND' in types else 'no'
        print(f"  {m.get('modelId', ''):<52} {status:<7} on-demand: {on_demand:<3}  {', '.join(types)}")
        detail = f"      {m.get('modelName', '')}  {m.get('modelArn', '')}"
        if m.get('modelId') in access:
            detail += f"\n      access: {access[m['modelId']]}"
        print(detail)


def print_profiles(profiles, every):
    print(f"{'Inference' if every else 'Claude inference'} profiles: {len(profiles)}")
    for p in sorted(profiles, key=lambda p: p.get('inferenceProfileId', '')):
        regions = sorted({arn_region(m.get('modelArn', '')) for m in p.get('models') or []} - {''})
        print(f"  {p.get('inferenceProfileId', ''):<52} {p.get('status', '?'):<7} "
              f"{p.get('type', ''):<14} {', '.join(regions)}")
        print(f"      {p.get('inferenceProfileName', '')}  {p.get('inferenceProfileArn', '')}")


def show(bedrock, region, args):
    models = bedrock.list_foundation_models(byProvider='Anthropic').get('modelSummaries', [])
    profiles = inference_profiles(bedrock)
    if not args.all_profiles:
        profiles = [p for p in profiles if is_claude(p)]
    access = {m['modelId']: model_access(bedrock, m['modelId']) for m in models} if args.access else {}
    if args.json:
        print(json.dumps({'region': region, 'models': models, 'profiles': profiles, 'access': access},
                         indent=2, default=str))
        return 0
    print_models(region, models, access)
    print()
    print_profiles(profiles, args.all_profiles)
    print()
    print(HINT)
    return 0


def probe(args, region):
    """Time tiny calls through the server's own code path; failures are logged by engine.llm."""
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING,
                        format='    %(levelname)s %(message)s')
    logging.getLogger('polaris.llm').setLevel(logging.DEBUG)  # one usage line per answered call
    llm = BedrockLLM(region=region)
    if not llm.enabled:
        return fail(llm.unavailable, 2)
    failed = 0
    for model in args.probe:
        print(f'{model} in {region}:', flush=True)
        latencies = []
        for i in range(1, max(1, args.repeat) + 1):
            started = time.monotonic()
            text = llm.complete(PROBE_SYSTEM, 'ping', max_tokens=args.max_tokens,
                                timeout=args.timeout, model=model)
            ms = (time.monotonic() - started) * 1000
            if text is None:
                print(f'  call {i}: failed after {ms:.0f} ms', flush=True)
            else:
                latencies.append(ms)
                print(f'  call {i}: {ms:.0f} ms -> {text.strip()[:40]!r}', flush=True)
        if latencies:
            print(f'  answered {len(latencies)}/{max(1, args.repeat)}: min {min(latencies):.0f} ms, '
                  f'median {statistics.median(latencies):.0f} ms')
        else:
            failed += 1
    return 1 if failed else 0


def main(argv=None):
    args = parse_args(argv)
    try:
        import boto3
        from botocore.exceptions import (BotoCoreError, ClientError, CredentialRetrievalError,
                                         NoCredentialsError, NoRegionError, PartialCredentialsError,
                                         ProfileNotFound, SSOError, TokenRetrievalError)
    except ImportError:
        return fail('boto3 is not installed: pip install -r requirements.txt', 2)
    credential_errors = (NoCredentialsError, PartialCredentialsError, ProfileNotFound, NoRegionError,
                         CredentialRetrievalError, TokenRetrievalError, SSOError)
    try:
        session = boto3.session.Session(region_name=args.region)
        if not session.region_name:
            return fail('no AWS region: pass --region or set AWS_REGION', 2)
        if session.get_credentials() is None:
            return fail(NO_CREDENTIALS, 2)
        if args.probe:
            return probe(args, session.region_name)
        return show(session.client('bedrock'), session.region_name, args)
    except ClientError as exc:
        return client_error(exc)
    except credential_errors as exc:
        return fail(f'{type(exc).__name__}: {exc}\n{NO_CREDENTIALS}', 2)
    except BotoCoreError as exc:
        return fail(f'{type(exc).__name__}: {exc}', 1)


if __name__ == '__main__':
    sys.exit(main())
