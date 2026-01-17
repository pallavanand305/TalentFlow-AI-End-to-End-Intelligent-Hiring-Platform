#!/bin/bash
# Security verification script - Run before pushing to GitHub

set -e

echo "🔒 TalentFlow AI - Security Verification"
echo "========================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
        ERRORS=$((ERRORS + 1))
    fi
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    WARNINGS=$((WARNINGS + 1))
}

echo "1. Checking .gitignore..."
if grep -q "^\.env$" .gitignore; then
    print_status 0 ".env is in .gitignore"
else
    print_status 1 ".env is NOT in .gitignore - CRITICAL!"
fi

echo ""
echo "2. Checking if .env is tracked..."
if git ls-files | grep -q "^\.env$"; then
    print_status 1 ".env is tracked in git - REMOVE IT!"
else
    print_status 0 ".env is not tracked"
fi

echo ""
echo "3. Checking for AWS credentials..."
if git ls-files | grep -qE "\.aws|credentials|\.pem$|\.key$"; then
    print_status 1 "AWS credential files found in git!"
else
    print_status 0 "No AWS credential files tracked"
fi

echo ""
echo "4. Checking for hardcoded secrets..."
if git grep -qiE "password\s*=\s*['\"][^'\"]{8,}['\"]" -- '*.py' 2>/dev/null; then
    print_warning "Potential hardcoded passwords found - verify they are test data"
else
    print_status 0 "No obvious hardcoded passwords"
fi

echo ""
echo "5. Checking for database files..."
if git ls-files | grep -qE "\.db$|\.sqlite$"; then
    print_status 1 "Database files tracked in git!"
else
    print_status 0 "No database files tracked"
fi

echo ""
echo "6. Checking for large files..."
LARGE_FILES=$(find . -type f -size +10M ! -path "./.git/*" ! -path "./venv/*" ! -path "./env/*" ! -path "./node_modules/*" 2>/dev/null | wc -l)
if [ "$LARGE_FILES" -gt 0 ]; then
    print_warning "Found $LARGE_FILES large files (>10MB)"
    find . -type f -size +10M ! -path "./.git/*" ! -path "./venv/*" ! -path "./env/*" ! -path "./node_modules/*" 2>/dev/null
else
    print_status 0 "No large files found"
fi

echo ""
echo "7. Checking .env.example..."
if [ -f .env.example ]; then
    if grep -qE "your-secret-key|change-in-production|example|placeholder" .env.example; then
        print_status 0 ".env.example contains placeholder values"
    else
        print_warning ".env.example might contain real values - verify manually"
    fi
else
    print_status 1 ".env.example not found!"
fi

echo ""
echo "8. Checking for private keys..."
if git grep -q "BEGIN.*PRIVATE KEY" -- '*.py' '*.txt' '*.pem' '*.key' 2>/dev/null; then
    print_status 1 "Private keys found in tracked files!"
else
    print_status 0 "No private keys in tracked files"
fi

echo ""
echo "9. Verifying important files exist..."
for file in README.md requirements.txt docker-compose.yml .gitignore; do
    if [ -f "$file" ]; then
        print_status 0 "$file exists"
    else
        print_status 1 "$file is missing!"
    fi
done

echo ""
echo "10. Checking git status..."
UNTRACKED=$(git status --porcelain | grep "^??" | wc -l)
if [ "$UNTRACKED" -gt 0 ]; then
    print_warning "$UNTRACKED untracked files - review before committing"
    git status --porcelain | grep "^??" | head -5
fi

echo ""
echo "========================================"
echo "Summary:"
echo "  Errors: $ERRORS"
echo "  Warnings: $WARNINGS"
echo ""

if [ $ERRORS -gt 0 ]; then
    echo -e "${RED}❌ FAILED - Fix errors before pushing!${NC}"
    exit 1
elif [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}⚠️  WARNINGS - Review warnings before pushing${NC}"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}✅ All checks passed! Safe to push.${NC}"
echo ""
echo "Next steps:"
echo "  git add ."
echo "  git commit -m \"Initial commit: TalentFlow AI\""
echo "  git push -u origin main"
