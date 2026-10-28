#!/usr/bin/env python3
"""
Fetch all PRs of modded-nanogpt (title + body + state) as raw material for the SPOD offline change library.

Why the list API instead of GET /pulls/{n} one by one:
  The unauthenticated GitHub core API is rate-limited to 60 req/hour. Fetching 350+ PRs
  individually would inevitably get blocked. The list API returns 100 per page and each
  object already includes body, so the full set needs only 4 requests.

Outputs data/prs.jsonl, one PR per line:
  number, title, body, state, merged(bool), created_at, closed_at, merged_at,
  user, labels, n_commits_url, html_url
  label_gold: 'merged' | 'closed_unmerged' | 'open'   <- coarse label, first-layer basis for pos/neg samples
"""
import json
import os
import time
import urllib.error
import urllib.request

REPO = "KellerJordan/modded-nanogpt"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUT = os.path.join(OUT_DIR, "prs.jsonl")
HEADERS = {"User-Agent": "spod-library-builder", "Accept": "application/vnd.github+json"}
# use a token if available, which raises the rate limit from 60/h to 5000/h (unused here, but kept as a hook)
if os.environ.get("GITHUB_TOKEN"):
    HEADERS["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]


def get(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as resp:
                remaining = resp.headers.get("X-RateLimit-Remaining")
                return json.loads(resp.read().decode("utf-8")), remaining
        except urllib.error.HTTPError as e:
            if e.code in (403, 429) and attempt < retries - 1:
                print(f"  rate-limited ({e.code}), sleep 60s and retry")
                time.sleep(60)
                continue
            raise


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    page = 1
    while True:
        url = (
            f"https://api.github.com/repos/{REPO}/pulls"
            f"?state=all&per_page=100&page={page}&sort=created&direction=asc"
        )
        batch, remaining = get(url)
        print(f"page {page}: {len(batch)} PRs (ratelimit remaining={remaining})")
        if not batch:
            break
        for p in batch:
            merged = p.get("merged_at") is not None
            if p["state"] == "open":
                label = "open"
            elif merged:
                label = "merged"
            else:
                label = "closed_unmerged"
            rows.append(
                {
                    "number": p["number"],
                    "title": p["title"],
                    "body": p.get("body") or "",
                    "state": p["state"],
                    "merged": merged,
                    "created_at": p["created_at"],
                    "closed_at": p.get("closed_at"),
                    "merged_at": p.get("merged_at"),
                    "user": (p.get("user") or {}).get("login"),
                    "labels": [l["name"] for l in p.get("labels", [])],
                    "html_url": p["html_url"],
                    "label_gold": label,
                }
            )
        if len(batch) < 100:
            break
        page += 1
        time.sleep(1)

    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # summary, printed directly to the terminal for easy checking
    from collections import Counter

    c = Counter(r["label_gold"] for r in rows)
    empty_body = sum(1 for r in rows if len(r["body"].strip()) == 0)
    short_body = sum(1 for r in rows if 0 < len(r["body"].strip()) < 80)
    print(f"\ntotal {len(rows)} items -> {OUT}")
    for k, v in c.most_common():
        print(f"  {k:16s} {v}")
    print(f"  empty body        {empty_body}")
    print(f"  body <80 chars    {short_body}")


if __name__ == "__main__":
    main()
