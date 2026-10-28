#!/usr/bin/env python3
"""
Fetch the data sources missed in the previous round, growing the library beyond 190 items.

The four missed blocks:
  1. issues (in GitHub, issues and PRs share one numbering space; the pulls API does not return issues)
     -> these contain idea proposals, failed-attempt reports, and discussions — the largest increment
  2. the record table of track_3_optimization (the previous round only parsed the main README's track 1/2)
  3. record.txt / writeup under each record directory (far more detailed than the one-line README entry)
  4. PR review comments (rejection reasons, for the reject semantic label) — needs a token; this script only fetches 1/2 for now

Uses the paginated list API; even unauthenticated it needs only a dozen or so requests.
"""
import json
import os
import time
import urllib.error
import urllib.request

REPO = "KellerJordan/modded-nanogpt"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
HEADERS = {"User-Agent": "spod-library-builder", "Accept": "application/vnd.github+json"}
if os.environ.get("GITHUB_TOKEN"):
    HEADERS["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]


def get(url, retries=3, raw=False):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read().decode("utf-8", errors="replace")
                return data if raw else json.loads(data)
        except urllib.error.HTTPError as e:
            if e.code in (403, 429) and attempt < retries - 1:
                print(f"  rate-limited ({e.code}), sleep 60s")
                time.sleep(60)
                continue
            print(f"  HTTP {e.code} on {url}")
            return None


def fetch_issues():
    """The issues API also returns PRs; filter them out via the pull_request field."""
    rows, page = [], 1
    while True:
        url = (f"https://api.github.com/repos/{REPO}/issues"
               f"?state=all&per_page=100&page={page}&sort=created&direction=asc")
        batch = get(url)
        if not batch:
            break
        pure = [x for x in batch if "pull_request" not in x]
        print(f"issues page {page}: {len(batch)} items, of which {len(pure)} are real issues")
        for x in pure:
            rows.append({
                "number": x["number"],
                "title": x["title"],
                "body": x.get("body") or "",
                "state": x["state"],
                "state_reason": x.get("state_reason"),
                "created_at": x["created_at"],
                "closed_at": x.get("closed_at"),
                "user": (x.get("user") or {}).get("login"),
                "labels": [l["name"] for l in x.get("labels", [])],
                "n_comments": x.get("comments", 0),
                "html_url": x["html_url"],
            })
        if len(batch) < 100:
            break
        page += 1
        time.sleep(1)
    out = os.path.join(OUT_DIR, "issues.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> {len(rows)} issues written to {out}")
    return rows


def fetch_extra_readmes():
    """README / writeup of track_3 and other record subdirectories."""
    cands = [
        "records/track_3_optimization/README.md",
        "records/track_2_medium/README.md",
        "records/track_1_short/README.md",
        "records/README.md",
    ]
    got = {}
    for path in cands:
        url = f"https://raw.githubusercontent.com/{REPO}/master/{path}"
        txt = get(url, raw=True)
        if txt:
            got[path] = txt
            print(f"  ✅ {path}  {len(txt)} chars  {txt.count(chr(10))} lines")
        else:
            print(f"  ✗  {path}")
        time.sleep(0.5)
    out = os.path.join(OUT_DIR, "extra_readmes.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(got, f, ensure_ascii=False, indent=1)
    print(f"-> {len(got)} files written to {out}")
    return got


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print("A) fetch issues")
    iss = fetch_issues()
    print("\n" + "=" * 70)
    print("B) fetch track_3 and other subdirectory READMEs")
    fetch_extra_readmes()

    print("\n" + "=" * 70)
    print("issue overview (titles, first 60)")
    for x in iss[:60]:
        print(f"  #{x['number']:<4d} [{x['state']:6s}] {len(x['body']):5d}ch c={x['n_comments']:<3d} {x['title'][:80]}")


if __name__ == "__main__":
    main()
