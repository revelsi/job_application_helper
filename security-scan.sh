#!/bin/bash

# Security Scanning Script for Job Application Helper
# Uses Semgrep to scan for vulnerabilities and security issues

set -e

echo "🔒 Starting Security Scan for Job Application Helper"
echo "=================================================="

# Check if semgrep is installed
if ! command -v semgrep &> /dev/null; then
    echo "❌ Semgrep not found. Installing..."
    pip install semgrep
fi

# Create security scan directory
mkdir -p security-reports

echo "📋 Scanning for security vulnerabilities..."
semgrep scan \
    --config=auto \
    --output=security-reports/security-vulnerabilities.json \
    --json \
    --verbose \
    backend/ frontend/ || true

echo "🔍 Scanning for Python-specific security issues..."
semgrep scan \
    --config=auto \
    --output=security-reports/python-security.json \
    --json \
    --verbose \
    backend/ || true

echo "⚛️ Scanning for TypeScript/JavaScript security issues..."
semgrep scan \
    --config=auto \
    --output=security-reports/typescript-security.json \
    --json \
    --verbose \
    frontend/ || true

echo "🔐 Scanning for authentication and authorization issues..."
semgrep scan \
    --config=.semgrep.yml \
    --output=security-reports/auth-security.json \
    --json \
    --verbose \
    backend/src/api/endpoints/keys.py \
    backend/src/core/credentials.py \
    frontend/src/lib/api.ts || true

echo "📁 Scanning for file upload vulnerabilities..."
semgrep scan \
    --config=.semgrep.yml \
    --output=security-reports/file-upload-security.json \
    --json \
    --verbose \
    backend/src/api/endpoints/documents.py || true

echo "🌐 Scanning for web security issues..."
semgrep scan \
    --config=.semgrep.yml \
    --output=security-reports/web-security.json \
    --json \
    --verbose \
    backend/src/api/main.py \
    frontend/src/ || true

echo "🔒 Scanning for secrets and credentials..."
semgrep scan \
    --config=auto \
    --output=security-reports/secrets-scan.json \
    --json \
    --verbose \
    backend/ frontend/ || true

echo "✅ Security scan completed!"
echo "📊 Reports saved in security-reports/ directory"

# Generate summary report
echo "📋 Generating summary report..."
python3 -c "
import json
import os
from datetime import datetime

def load_findings(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            return data.get('results', [])
    return []

# Load all findings
all_findings = []
for report in ['security-vulnerabilities.json', 'python-security.json', 
               'typescript-security.json', 'auth-security.json', 
               'file-upload-security.json', 'web-security.json',
               'secrets-scan.json']:
    findings = load_findings(f'security-reports/{report}')
    all_findings.extend(findings)

# Categorize by severity
severity_counts = {'ERROR': 0, 'WARNING': 0, 'INFO': 0}
for finding in all_findings:
    severity = finding.get('extra', {}).get('severity', 'INFO')
    severity_counts[severity] += 1

# Generate summary
summary = {
    'scan_date': datetime.now().isoformat(),
    'total_findings': len(all_findings),
    'severity_breakdown': severity_counts,
    'critical_findings': [f for f in all_findings if f.get('extra', {}).get('severity') == 'ERROR'],
    'high_findings': [f for f in all_findings if f.get('extra', {}).get('severity') == 'WARNING']
}

with open('security-reports/scan-summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'📊 Scan Summary:')
print(f'   Total Findings: {len(all_findings)}')
print(f'   Critical (ERROR): {severity_counts[\"ERROR\"]}')
print(f'   High (WARNING): {severity_counts[\"WARNING\"]}')
print(f'   Info: {severity_counts[\"INFO\"]}')

if severity_counts['ERROR'] > 0:
    print('🚨 CRITICAL: Found security vulnerabilities that need immediate attention!')
    exit(1)
elif severity_counts['WARNING'] > 0:
    print('⚠️  WARNING: Found potential security issues that should be reviewed.')
else:
    print('✅ No critical security issues found!')
"

echo ""
echo "🔍 To view detailed findings, check the JSON files in security-reports/"
echo "📋 For a quick overview, see security-reports/scan-summary.json" 