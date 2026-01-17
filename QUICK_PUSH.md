# ⚡ Quick Push Commands

## 🔒 Step 1: Security Check (REQUIRED)

```bash
# Linux/Mac:
chmod +x scripts/verify_before_push.sh
./scripts/verify_before_push.sh

# Windows (PowerShell):
powershell -ExecutionPolicy Bypass -File scripts/verify_before_push.ps1
```

**⚠️ Only proceed if all checks pass!**

---

## 🚀 Step 2: Push to GitHub

### Option A: Using GitHub CLI (Recommended)

```bash
# Login to GitHub
gh auth login

# Create repository and push
gh repo create TalentFlow-AI --public --description "AI-powered intelligent hiring platform with ML-based candidate screening"

# Push code
git branch -M main
git push -u origin main
```

### Option B: Manual Setup

```bash
# 1. Create repository on GitHub.com
# Go to: https://github.com/new
# Name: TalentFlow-AI
# Description: AI-powered intelligent hiring platform with ML-based candidate screening
# Public/Private: Your choice
# DO NOT initialize with README

# 2. Add remote and push
git remote add origin https://github.com/YOUR_USERNAME/TalentFlow-AI.git
git branch -M main
git push -u origin main
```

---

## ✅ Step 3: Verify

```bash
# Check remote
git remote -v

# View on GitHub
gh repo view --web
# Or visit: https://github.com/YOUR_USERNAME/TalentFlow-AI
```

---

## 🎉 Done!

Your code is now on GitHub!

**Next steps:**
1. Review README on GitHub
2. Add repository topics
3. Share with recruiters
4. Continue development

**Need detailed instructions?** See `PUSH_TO_GITHUB.md`
