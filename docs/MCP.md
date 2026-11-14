# Use HirePanel from an AI assistant (MCP)

MCP (Model Context Protocol) is a standard way to give an AI assistant tools. This server gives Claude Desktop, or any MCP client, five tools that talk to your HirePanel API.

## What it can and cannot do

| Tool | What it does |
|---|---|
| `check_connection` | Checks the URL and key work. |
| `list_jobs` | Lists jobs with their ids. |
| `rank_candidates` | Ranks candidates for one job. Shows score, confidence, AI recommendation, safety flags. |
| `get_candidate` | Explains one candidate: the quote behind each requirement, whether the quote was verified, flags, suggested interview questions. |
| `score_resume` | Scores pasted resume text and returns the explanation. Saves it like the paste box in the app. Turn off with `HIREPANEL_MCP_READ_ONLY=true`. |

There is **no** tool to approve, reject, email, export or delete. A test (`tests/test_mcp.py`) fails if one is ever added.

## Safety choices

- **Names masked** ("Priya Sharma" becomes "P. S.") unless you set `HIREPANEL_MCP_REVEAL_NAMES=true`.
- **Never returned:** email, phone, links, raw resume text, drafted emails.
- **Prompt-injection aware.** Resume-derived text is stripped of invisible and control characters, shortened, and every reply starts with a notice that it is untrusted data. The server's instructions tell the assistant never to follow instructions found inside it. This reduces the risk; it does not remove it, because the assistant is still a language model reading attacker-controlled text.
- **Least privilege.** The server only holds the key you give it in `HIREPANEL_API_KEY`. It does not read your `.env`. Without a key it works against a `DEMO_MODE` site for reading only.
- **Ids are validated** before any request is built.

## Settings (environment variables)

| Name | Meaning | Default |
|---|---|---|
| `HIREPANEL_URL` | Where HirePanel runs | `http://127.0.0.1:8000` |
| `HIREPANEL_API_KEY` | Your `API_KEY` (leave empty for the public read-only demo) | empty |
| `HIREPANEL_MCP_READ_ONLY` | `true` hides `score_resume` | `false` |
| `HIREPANEL_MCP_REVEAL_NAMES` | `true` shows full names | `false` |

## Install

```
pip install -r requirements-mcp.txt
```

Claude Desktop: Settings, Developer, Edit Config. Add this to `claude_desktop_config.json` (change the paths to yours; use double backslashes on Windows):

```json
{
  "mcpServers": {
    "hirepanel": {
      "command": "D:\\hirepanel\\.venv\\Scripts\\python.exe",
      "args": ["D:\\hirepanel\\mcp_server\\run.py"],
      "env": {
        "HIREPANEL_URL": "http://127.0.0.1:8000",
        "HIREPANEL_API_KEY": "paste-your-API_KEY-here"
      }
    }
  }
}
```

Fully quit and reopen Claude Desktop. Then ask: "Use HirePanel to list my jobs, then rank the candidates and explain the top one."

That config file holds your key in plain text on your computer. Never put it in git or share it.

## Test without Claude Desktop

```
python -m pytest tests/test_mcp.py -q
```

## Honest limits

- It is a thin layer over the REST API; it adds no scoring logic of its own.
- `score_resume` saves the resume to your database. Use a throwaway job for experiments.
- An assistant can still be tricked by hostile resume text. Keep a human in the loop, as HirePanel does.
