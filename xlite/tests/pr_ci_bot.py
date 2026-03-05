#!/usr/bin/env python3
import requests
import subprocess
import argparse
from pathlib import Path

GITCODE_API_BASE = "https://gitcode.com/api/v5"
REPO_OWNER = "openeuler"
REPO_NAME = "GVirt"
TOKEN = None
XLITE_DIR = "/workspaces/code/opencode/GVirt"
VLLM_ASCEND = "/workspaces/code/opencode/vllm-ascend"
VLLM_DIR = "/workspaces/code/opencode/vllm"
STATIC_CHECK_CMD = "cd {}/xlite && python tests/run_static_checks.py 2>&1".format(XLITE_DIR)
UT_TEST_CMD = "cd {}/xlite && bash tests/run_accuracy.sh".format(XLITE_DIR)

GIT_REPO_DIR = XLITE_DIR

def get_open_prs():
    url = f"{GITCODE_API_BASE}/repos/{REPO_OWNER}/{REPO_NAME}/pulls"
    params = {"state": "open", "per_page": 100}
    headers = {"Authorization": f"Bearer {TOKEN}"}
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching PRs: {e}")
        return []

def get_pr_comments(pr_number):
    url = f"{GITCODE_API_BASE}/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}/comments"
    headers = {"Authorization": f"Bearer {TOKEN}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching comments for PR #{pr_number}: {e}")
        return []

def has_ci_bot_comment(comments):
    for comment in comments:
        body = comment.get("body", "")
        if "CI Test Results" in body or "CI测试结果" in body:
            return True
    return False

def add_remote_if_not_exists(remote_name, remote_url):
    try:
        result = subprocess.run(["git", "remote", "add", remote_name, remote_url], capture_output=True, text=True, cwd=GIT_REPO_DIR)
        if result.returncode != 0 and "already exists" not in result.stderr:
            print(f"Warning: {result.stderr}")
    except Exception as e:
        print(f"Error adding remote: {e}")

def fetch_pr_branch(remote_name, branch_name, local_branch_name):
    try:
        print(f"Current directory: {GIT_REPO_DIR}")
        current_branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, cwd=GIT_REPO_DIR, timeout=300).stdout.strip()
        print(f"Current branch: {current_branch}")
        if current_branch == local_branch_name:
            print("Switching to master branch")
            subprocess.run(["git", "checkout", "master"], capture_output=True, text=True, cwd=GIT_REPO_DIR, timeout=60)
        print(f"Deleting local branch {local_branch_name} if exists")
        subprocess.run(["git", "branch", "-D", local_branch_name], capture_output=True, text=True, cwd=GIT_REPO_DIR, timeout=30)
        cmd = ["git", "fetch", remote_name, f"{branch_name}:{local_branch_name}"]
        print(f"Fetching {remote_name}/{branch_name} to {local_branch_name}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=GIT_REPO_DIR, timeout=300)
        if result.returncode != 0:
            print(f"Error fetching branch: {result.stderr}")
            return False
        print(f"Successfully fetched branch")
        return True
    except subprocess.TimeoutExpired:
        print(f"Timeout while fetching branch {branch_name}")
        return False
    except Exception as e:
        print(f"Error fetching PR: {e}")
        return False

def checkout_branch(branch_name):
    try:
        print(f"Stashing changes...")
        subprocess.run(["git", "stash", "-u"], capture_output=True, text=True, cwd=GIT_REPO_DIR, timeout=60)
        cmd = ["git", "checkout", branch_name]
        print(f"Checking out branch {branch_name}...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=GIT_REPO_DIR, timeout=60)
        if result.returncode != 0:
            if "untracked working tree files" in result.stderr:
                print("Cleaning untracked files...")
                subprocess.run(["git", "clean", "-fd"], capture_output=True, text=True, cwd=GIT_REPO_DIR, timeout=30)
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=GIT_REPO_DIR, timeout=60)
            if result.returncode != 0:
                print(f"Error checking out branch: {result.stderr}")
                return False
        print(f"Successfully checked out branch {branch_name}")
        return True
    except subprocess.TimeoutExpired:
        print(f"Timeout while checking out branch {branch_name}")
        return False
    except Exception as e:
        print(f"Error checking out branch: {e}")
        return False

def delete_branch(branch_name):
    try:
        print(f"Switching to master branch before deleting...")
        subprocess.run(["git", "checkout", "master"], capture_output=True, text=True, cwd=GIT_REPO_DIR, timeout=60)
        print(f"Deleting branch {branch_name}...")
        result = subprocess.run(["git", "branch", "-D", branch_name], capture_output=True, text=True, cwd=GIT_REPO_DIR, timeout=30)
        if result.returncode == 0:
            print(f"Successfully deleted branch {branch_name}")
            return True
        else:
            print(f"Failed to delete branch {branch_name}: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"Timeout while deleting branch {branch_name}")
        return False
    except Exception as e:
        print(f"Error deleting branch {branch_name}: {e}")
        return False

def run_static_check():
    try:
        print(f"Running command: {STATIC_CHECK_CMD}")
        cmd = ["bash", "-c", STATIC_CHECK_CMD]
        print("Starting static check (timeout: 30 minutes)...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        print(f"Static check completed with return code: {result.returncode}")
        return result.stdout, result.returncode
    except subprocess.TimeoutExpired:
        print("Static check: timeout after 30 minutes")
        return "Error: Static check timed out", 1
    except Exception as e:
        print(f"Error running static check: {e}")
        return f"Error: {e}", 1

def run_vllm_ascend_static_check():
    try:
        mypy_cmd = f"MYPYPATH={VLLM_DIR}:{XLITE_DIR}/xlite mypy --follow-imports skip --check-untyped-defs --python-version 3.11 {VLLM_ASCEND}/vllm_ascend/xlite/xlite.py"
        print(f"Running vllm_ascend static check: {mypy_cmd}")
        cmd = ["bash", "-c", mypy_cmd]
        print("Starting vllm_ascend static check (timeout: 30 minutes)...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        print(f"Vllm_ascend static check completed with return code: {result.returncode}")
        return result.stdout, result.returncode
    except subprocess.TimeoutExpired:
        print("Vllm_ascend static check: timeout after 30 minutes")
        return "Error: Vllm_ascend static check timed out", 1
    except Exception as e:
        print(f"Error running vllm_ascend static check: {e}")
        return f"Error: {e}", 1

def parse_check_results(output):
    lines = output.split('\n')
    cpp_status = "Unknown"
    python_status = "Unknown"
    
    for line in lines:
        if "CPP" in line and ("PASSED" in line or "FAILED" in line):
            if "PASSED" in line:
                cpp_status = "Passed"
            else:
                cpp_status = "Failed"
        elif "PYTHON" in line and ("PASSED" in line or "FAILED" in line):
            if "PASSED" in line:
                python_status = "Passed"
            else:
                python_status = "Failed"
    
    all_passed = cpp_status == "Passed" and python_status == "Passed"
    return {"cpp": cpp_status, "python": python_status, "success": all_passed}

def parse_vllm_ascend_static_check_results(output, returncode):
    return {"success": returncode == 0}

def run_ut_test():
    try:
        xlite_path = f"{XLITE_DIR}/xlite"
        print(f"Building xlite {xlite_path}...")
        build_cmd = ["bash", "-c", "rm -rf build && mkdir -p build && cmake -B build && cmake --build build -j && cmake --install build"]
        build_result = subprocess.run(build_cmd, capture_output=True, text=True, cwd=xlite_path, timeout=1200)
        if build_result.returncode != 0:
            print(f"Failed to build xlite: {build_result.stderr}")
            return f"Error building: {build_result.stderr}", 1
        print("xlite built and installed successfully")
        
        print(f"Running command: {UT_TEST_CMD}")
        cmd = ["bash", "-c", UT_TEST_CMD]
        print("Starting UT test (timeout: 20 minutes)...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=xlite_path, timeout=1200)
        print(f"UT test completed with return code: {result.returncode}")
        
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        
        if "All accuracy tests PASSED!" in output:
            return output, 0
        else:
            return output, 1
    except subprocess.TimeoutExpired:
        print("UT test: timeout after 20 minutes")
        return "Error: UT test timed out", 1
    except Exception as e:
        print(f"Error running test: {e}")
        return f"Error: {e}", 1

def post_pr_comment(pr_number, xlite_check_results, xlite_output, vllm_ascend_check_results, vllm_ascend_output, ut_output=None, ut_returncode=None):
    url = f"{GITCODE_API_BASE}/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}/comments"
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    
    xlite_status = "Failed" if not xlite_check_results["success"] else "Passed"
    vllm_ascend_status = "Failed" if not vllm_ascend_check_results["success"] else "Passed"
    
    static_check_passed = xlite_check_results["success"] and vllm_ascend_check_results["success"]
    
    xlite_suggestion = ""
    if not xlite_check_results["success"]:
        xlite_suggestion = f"\n\n**Suggestion**: Please run `python xlite/tests/run_static_checks.py` to see detailed error messages and fix them."

    vllm_ascend_suggestion = ""
    if not vllm_ascend_check_results["success"]:
        vllm_ascend_suggestion = f"\n\n**Suggestion**: Please run `mypy --follow-imports skip --check-untyped-defs --python-version 3.11 vllm_ascend/xlite/xlite.py` to see detailed error messages and fix them."

    body = f"""### CI Test Results

## 1. GVirt/xlite Static Check
**Status**: {xlite_status}

**Check Results**:
- C++: {xlite_check_results['cpp']}
- Python: {xlite_check_results['python']}
{xlite_suggestion}

<details>
<summary>Click to see detailed output</summary>

```
{xlite_output}
```

</details>

## 2. vllm-ascend Static Check
**Status**: {vllm_ascend_status}
{vllm_ascend_suggestion}

<details>
<summary>Click to see detailed output</summary>

```
{vllm_ascend_output}
```

</details>
"""
    
    if ut_output is not None:
        ut_status = "Passed" if ut_returncode == 0 else "Failed"
        body += f"""

## 3. UT Test
**Status**: {ut_status}

<details>
<summary>Click to see detailed output</summary>

```
{ut_output}
```

</details>
"""
    
    body += """
---
*This comment was automatically generated by CI bot*"""
    
    try:
        response = requests.post(url, json={"body": body}, headers=headers)
        response.raise_for_status()
        print(f"Successfully posted comment to PR #{pr_number}")
        return True
    except requests.RequestException as e:
        print(f"Error posting comment to PR #{pr_number}: {e}")
        return False

def process_pr(pr):
    pr_number = pr["number"]
    pr_title = pr["title"]
    source_branch = pr["source_branch"]
    source_repo = pr["head"]["repo"]
    source_owner = source_repo["owner"]["login"]
    print(f"\n{'='*60}")
    print(f"Processing PR #{pr_number}: {pr_title}")
    print(f"{'='*60}")
    comments = get_pr_comments(pr_number)
    if has_ci_bot_comment(comments):
        print(f"PR #{pr_number} already has CI bot comment, skipping...")
        return
    print(f"Fetching PR #{pr_number} branch...")
    remote_name = source_owner
    remote_url = f"https://gitcode.com/{source_owner}/{REPO_NAME}.git"
    local_branch = f"pr-{pr_number}"
    add_remote_if_not_exists(remote_name, remote_url)
    if not fetch_pr_branch(remote_name, source_branch, local_branch):
        print(f"Failed to fetch PR #{pr_number}")
        return
    if not checkout_branch(local_branch):
        print(f"Failed to checkout PR #{pr_number}")
        return
    
    print(f"Running GVirt/xlite static check: {STATIC_CHECK_CMD}")
    xlite_output, xlite_returncode = run_static_check()
    xlite_check_results = parse_check_results(xlite_output)
    print(f"GVirt/xlite check results: C++ {xlite_check_results['cpp']}, Python {xlite_check_results['python']}")
    
    print(f"Running vllm-ascend static check for PR #{pr_number}...")
    vllm_ascend_output, vllm_ascend_returncode = run_vllm_ascend_static_check()
    vllm_ascend_check_results = parse_vllm_ascend_static_check_results(vllm_ascend_output, vllm_ascend_returncode)
    print(f"Vllm-ascend static check results: {'Passed' if vllm_ascend_check_results['success'] else 'Failed'}")
    
    static_check_passed = xlite_check_results["success"] and vllm_ascend_check_results["success"]
    
    ut_output = None
    ut_returncode = None
    
    if static_check_passed:
        print(f"Static checks passed, running UT test for PR #{pr_number}...")
        ut_output, ut_returncode = run_ut_test()
        print(f"UT test results: {'Passed' if ut_returncode == 0 else 'Failed'}")
    else:
        print(f"Static checks failed, skipping UT test for PR #{pr_number}...")
    
    post_pr_comment(pr_number, xlite_check_results, xlite_output, vllm_ascend_check_results, vllm_ascend_output, ut_output, ut_returncode)
    
    print(f"Deleting test branch {local_branch}...")
    delete_branch(local_branch)

def main():
    global TOKEN, XLITE_DIR, VLLM_ASCEND, VLLM_DIR, STATIC_CHECK_CMD, GIT_REPO_DIR, UT_TEST_CMD
    parser = argparse.ArgumentParser(description='PR CI Bot')
    parser.add_argument('--token', required=True, help='GitCode API token')
    parser.add_argument('--xlite-dir', default=XLITE_DIR, help='Path to GVirt/xlite directory')
    parser.add_argument('--vllm-ascend-dir', default=VLLM_ASCEND, help='Path to vllm-ascend directory')
    parser.add_argument('--vllm-dir', default=VLLM_DIR, help='Path to vllm directory')
    args = parser.parse_args()
    TOKEN = args.token
    XLITE_DIR = args.xlite_dir
    VLLM_ASCEND = args.vllm_ascend_dir
    VLLM_DIR = args.vllm_dir
    STATIC_CHECK_CMD = "cd {}/xlite && python tests/run_static_checks.py 2>&1".format(XLITE_DIR)
    UT_TEST_CMD = "cd {}/xlite && bash tests/run_accuracy.sh".format(XLITE_DIR)
    GIT_REPO_DIR = XLITE_DIR
    
    print("Starting PR CI Bot")
    print(f"Repository: {REPO_OWNER}/{REPO_NAME}")
    print("\nFetching open PRs...")
    prs = get_open_prs()
    if not prs:
        print("No open PRs found")
        return
    print(f"Found {len(prs)} open PR(s)")
    for pr in prs:
        process_pr(pr)
    print(f"\n{'='*60}")
    print("All PRs processed successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()