# graphweave

Ask a question about any public GitHub repository and see the answer as a map of the pull requests, issues, commits and people behind it.

Everything runs on your computer. graphweave only talks to GitHub, to read the repository's recent activity.

## What you need

- Python 3.12 or newer. Check with `python --version`.
- The name of a public GitHub repository, written as `owner/name`, for example `psf/requests`.

## Get started

**1. Install graphweave**

```bash
pip install grag-trace-viewer
```

**2. Ask a question**

Put the repository first, then your question in quotes:

```bash
graphrag-github-trace psf/requests "What changed in how sessions handle retries?"
```

graphweave reads the repository's latest pull requests, issues and commits, prints the ones that best match your question, and saves the full result to `graphrag_out/trace_state.json`.

**3. Open the map**

```bash
graphrag graphrag_out/trace_state.json
```

Your browser opens at `http://127.0.0.1:4630`. Click anything on the map to see its details. The map is served from your own computer, so nobody else can see it. Press `Ctrl+C` in the terminal to close it.

## Good questions to ask

graphweave answers from a repository's recent activity, so questions about who did what, and what changed, work best:

- `"Who has been working on the test suite?"`
- `"What changed in how errors are reported?"`
- `"Which pull requests fixed issues about timeouts?"`
- `"What is being done about Windows support?"`

## Add a GitHub token (recommended)

Without a token, GitHub allows 60 requests an hour, which is enough for a couple of questions. With a token you get 5,000.

1. Go to [github.com/settings/tokens](https://github.com/settings/tokens) and choose **Generate new token**. For public repositories it needs no extra permissions.
2. Set it in the terminal you are using.

macOS and Linux:

```bash
export GITHUB_TOKEN=paste_your_token_here
```

Windows (PowerShell):

```powershell
$env:GITHUB_TOKEN="paste_your_token_here"
```

## Look further back

Add any of these to the command in step 2:

| Add | What it does |
| --- | --- |
| `--prs 50` | Read the last 50 pull requests (default 15) |
| `--issues 50` | Read the last 50 issues (default 15) |
| `--commits 50` | Read the last 50 commits (default 20) |
| `--reviews` | Include who reviewed each pull request |
| `--files` | Include which files each pull request changed |
| `--source` | Also read up to 20 source files from the repository |
| `--out result.json` | Save the result to a different file |

Reading more uses more of your GitHub allowance, so add a token first.

## Keep several questions

Save each question to its own file in one folder:

```bash
graphrag-github-trace psf/requests "Who worked on proxies?" --out my-questions/proxies.json
graphrag-github-trace psf/requests "What changed in SSL handling?" --out my-questions/ssl.json
```

Then open the whole folder and switch between them:

```bash
graphrag my-questions/
```

## If something goes wrong

| You see | Do this |
| --- | --- |
| A message about GitHub's rate limit | Add a token, or wait an hour |
| The browser doesn't open | Copy the address printed in the terminal into your browser |
| "Address already in use" | Use another port: `graphrag graphrag_out/trace_state.json --port 4700` |
| `graphrag` is not recognized | Close and reopen the terminal after installing. If it still fails, make sure Python's scripts folder is on your PATH |
