# 🔒 Security Checklist - Before Pushing to GitHub

## ✅ Pre-Push Security Verification

Run this checklist **BEFORE** pushing to GitHub to ensure no sensitive data is committed.

### 1. Environment Variables Check

```bash
# Verify .env is in .gitignore
grep -q "^\.env$" .gitignore && echo "✅ .env is ignored" || echo "❌ WARNING: .env not in .gitignore!"

# Check if .env exists (it should NOT be committed)
git ls-files | grep -q "^\.env$" && echo "❌ DANGER: .env is tracked!" || echo "✅ .env is not tracked"

# Verify .env.example has no real secrets
grep -i "password\|secret\|key" .env.example
# Should only show placeholder values like "your-secret-key-change-in-production"
```

### 2. Secrets and Credentials Check

```bash
# Search for potential secrets in tracked files
git grep -i "password\s*=\s*['\"]" -- '*.py' '*.js' '*.yml' '*.yaml' '*.json'
git grep -i "api_key\s*=\s*['\"]" -- '*.py' '*.js' '*.yml' '*.yaml' '*.json'
git grep -i "secret_key\s*=\s*['\"]" -- '*.py' '*.js' '*.yml' '*.yaml' '*.json'
git grep -i "aws_access_key" -- '*.py' '*.js' '*.yml' '*.yaml' '*.json'

# Should return no results or only placeholder values
```

### 3. AWS Credentials Check

```bash
# Verify no AWS credentials are tracked
git ls-files | grep -E "\.aws|credentials|\.pem|\.key$"
# Should return nothing

# Check for AWS keys in code
git grep -E "AKIA[0-9A-Z]{16}" -- '*.py' '*.js' '*.yml' '*.yaml' '*.json'
# Should return nothing
```

### 4. Database Credentials Check

```bash
# Check for hardcoded database passwords
git grep -i "postgres://.*:.*@" -- '*.py' '*.yml' '*.yaml'
# Should only show examples with "postgres:postgres@localhost" or environment variables

# Verify no database files are tracked
git ls-files | grep -E "\.db$|\.sqlite$"
# Should return nothing
```

### 5. Private Keys and Certificates Check

```bash
# Check for private keys
git ls-files | grep -E "\.pem$|\.key$|\.crt$|\.cer$|\.p12$|\.pfx$"
# Should return nothing

# Search for BEGIN PRIVATE KEY in files
git grep "BEGIN.*PRIVATE KEY" -- '*.py' '*.txt' '*.pem' '*.key'
# Should return nothing
```

### 6. Configuration Files Check

```bash
# Verify important files are present
ls -la .env.example
ls -la .gitignore
ls -la README.md

# Verify .env is NOT present in git
git ls-files | grep "^\.env$"
# Should return nothing
```

### 7. Large Files Check

```bash
# Check for large files that shouldn't be committed
find . -type f -size +10M ! -path "./.git/*" ! -path "./venv/*" ! -path "./node_modules/*"
# Should return nothing or only expected large files

# Check git objects for large files
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  awk '/^blob/ {print substr($0,6)}' | \
  sort --numeric-sort --key=2 | \
  tail -n 10
```

### 8. Sensitive Data Patterns Check

```bash
# Check for email addresses (should only be examples)
git grep -E "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}" -- '*.py' | grep -v "example\|test\|@talentflow"

# Check for phone numbers
git grep -E "\+?[0-9]{10,}" -- '*.py' | grep -v "test\|example"

# Check for credit card patterns
git grep -E "[0-9]{4}[- ]?[0-9]{4}[- ]?[0-9]{4}[- ]?[0-9]{4}" -- '*.py'
```

### 9. Docker Secrets Check

```bash
# Verify docker-compose.yml uses environment variables
grep -E "password:|secret:" docker-compose.yml
# Should use ${VARIABLE} syntax or default values like "postgres"

# Check for hardcoded secrets in Dockerfiles
git grep -i "password\|secret\|key" -- Dockerfile* docker/*.Dockerfile
```

### 10. Test Data Check

```bash
# Verify no real user data in tests
git grep -E "real.*email|actual.*password" -- tests/
# Should return nothing

# Check for test fixtures with sensitive data
find tests/ -name "*.json" -o -name "*.yaml" -o -name "*.yml" | xargs grep -i "password\|secret"
```

## 🛡️ Security Best Practices Verified

- ✅ `.env` file is in `.gitignore`
- ✅ `.env.example` contains only placeholder values
- ✅ No AWS credentials in code
- ✅ No database passwords hardcoded
- ✅ No private keys or certificates
- ✅ All secrets use environment variables
- ✅ No large binary files
- ✅ No real user data in tests
- ✅ Docker secrets use environment variables
- ✅ Configuration files use placeholders

## 🚨 If You Find Issues

### Remove Sensitive File from Git

```bash
# If you accidentally added a sensitive file
git rm --cached .env
git commit -m "Remove sensitive file"

# If already pushed, you need to rewrite history (DANGEROUS)
# Contact your team before doing this!
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env" \
  --prune-empty --tag-name-filter cat -- --all
```

### Remove Sensitive Data from File

```bash
# If you committed a file with secrets
# 1. Remove the secrets from the file
# 2. Commit the cleaned version
git add filename
git commit -m "Remove sensitive data from filename"

# 3. If already pushed, consider the secrets compromised
# - Rotate all exposed credentials immediately
# - Update all affected services
```

## 📋 Final Checklist

Before running `git push`:

- [ ] Ran all security checks above
- [ ] No sensitive data found
- [ ] `.env` is not tracked
- [ ] `.env.example` has only placeholders
- [ ] All secrets use environment variables
- [ ] No AWS credentials in code
- [ ] No private keys committed
- [ ] No large unnecessary files
- [ ] Docker configs use env vars
- [ ] Test data is synthetic only

## ✅ Ready to Push!

If all checks pass, you're safe to push:

```bash
git status
git add .
git commit -m "Initial commit: TalentFlow AI"
git push -u origin main
```

## 🆘 Emergency: Secrets Already Pushed

If you've already pushed secrets to GitHub:

1. **Immediately rotate all exposed credentials**
2. **Remove secrets from repository history**
3. **Force push the cleaned history**
4. **Notify your team**
5. **Review GitHub's guide**: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository

## 📚 Additional Resources

- [GitHub: Removing sensitive data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [Git-secrets tool](https://github.com/awslabs/git-secrets)
- [Gitleaks](https://github.com/gitleaks/gitleaks)
- [TruffleHog](https://github.com/trufflesecurity/trufflehog)

---

**Remember**: Once pushed to GitHub, consider any secrets as compromised. Always verify before pushing!
