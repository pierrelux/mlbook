---
name: commit-push-publish
description: >
  Commit all changes, push to remote, and publish the book to GitHub Pages.
  Use when the user invokes /commit-push-publish.
user_invocable: true
---

# Commit, Push, and Publish

Run these three steps in sequence for the mlbook project:

1. **Commit**: Stage all modified and new files, then create a commit with a descriptive message summarizing the changes. Follow the repository's commit conventions (see git log for style). End the commit message with the Co-Authored-By trailer.

2. **Push**: Push the commit to the remote (`git push`).

3. **Publish**: Run `source publish.sh` from the project root (`/Users/pierre-luc.bacon/Documents/mlbook/`) to build the site with execution and deploy to GitHub Pages.

Important:
- Before committing, run `git status` and `git diff` to understand what changed.
- Do not commit files that contain secrets (.env, credentials, etc.).
- If any step fails, stop and report the error to the user.
