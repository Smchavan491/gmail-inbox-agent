# Gmail Inbox Agent

An MCP (Model Context Protocol) server that turns Gmail into a set of tools an AI agent can reason over — classifying emails by real urgency, catching important messages before they get buried, and drafting clean replies from rough notes. Includes a custom orchestration loop with human-approval gating on any action that changes your inbox.

## Why this exists

Gmail's built-in filters can only match keywords and senders — they can't judge whether an email is genuinely urgent versus just containing similar words to spam, can't draft a reply from a rough note, and have no "needs a reply" tracking. This project adds an LLM-reasoning layer on top of real Gmail data to do exactly that, while leaving anything reversible (starring, archiving) low-risk and anything irreversible (sending) entirely human-approved.

## What it does

- Classifies incoming email by urgency and type (8 categories: Urgent, Important, Normal, Promotional, Social, Updates, Forums, Purchases) using an LLM, not keyword rules
- Surfaces important emails that would otherwise get missed among lower-priority mail
- Drafts clean, grammatically correct emails from rough/typo-filled input — saved as a Gmail draft only, never auto-sent
- Supports two ways to use it: conversationally through Claude Desktop, or through a custom agent script with a multi-step reasoning loop and human approval gating on write actions

## Tools exposed

| Tool | Type | Description |
|---|---|---|
| `list_recent_emails` | Read | Fetch recent inbox emails (ID, sender, subject) |
| `classify_email` | LLM reasoning | Classify one email into an 8-category taxonomy with a reason |
| `find_important_unstarred` | Multi-step reasoning | Scan recent emails, classify each, return unstarred Urgent/Important ones |
| `star_email` | Write | Star a specific email |
| `list_starred_emails` | Read | List currently starred emails |
| `list_sent_emails` | Read | List emails you've sent |
| `archive_email` | Write | Archive a specific email |
| `list_archived_emails` | Read | List archived emails |
| `list_spam_emails` | Read | List emails in Spam |
| `list_trash_emails` | Read | List emails in Trash |
| `create_draft` | LLM generation + Write | Turn rough text into a clean draft, saved (never sent) |
| `get_draft` | Read | Fetch the full text of a specific draft |
| `list_drafts` | Read | List all current drafts |

## Architecture

```
Gmail account
     │  OAuth (read/modify scope)
     ▼
gmail_mcp_server.py  ── MCP tools, each a thin wrapper around
     │                  the Gmail API + (for reasoning tools) an LLM call
     │
     ├── Claude Desktop  ── conversational use, Anthropic's own
     │                      orchestration decides which tools to call
     │
     └── agent.py         ── custom orchestration loop:
                             task → model picks a tool → run it →
                             feed result back → repeat until done.
                             Write actions pause for human approval.
```

## Tech stack

- **Language / env**: Python, managed with `uv`
- **MCP**: official `mcp` Python SDK (`FastMCP`)
- **Gmail access**: `google-api-python-client` + `google-auth-oauthlib` (OAuth 2.0, `gmail.modify` scope)
- **LLM**: Groq API (`openai/gpt-oss-20b`), accessed via the OpenAI-compatible client — used for both classification (structured judgment) and generation (draft writing)
- **Storage**: none needed beyond Gmail itself — no local database
- **Secrets**: `.env` (API keys) + OAuth `credentials.json`/`token.json`, all excluded from version control

## Setup

### 1. Clone and install dependencies

```bash
git clone <this-repo>
cd gmail-inbox-agent
uv sync
```

### 2. Gmail API credentials

1. Create a project in [Google Cloud Console](https://console.cloud.google.com)
2. Enable the Gmail API
3. Configure the OAuth consent screen (External, add yourself as a test user)
4. Under **Data Access**, add the scope `https://www.googleapis.com/auth/gmail.modify`
5. Create an OAuth Client ID (type: Desktop app), download it, and save it as `credentials.json` in the project root

### 3. LLM API key

Sign up at [console.groq.com](https://console.groq.com) (free) and generate an API key.

### 4. Environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your-groq-key-here
```

### 5. First run (triggers Gmail login)

```bash
uv run mcp dev gmail_mcp_server.py
```

This opens the MCP Inspector and, on first use of any tool, opens a browser window to complete Gmail OAuth. A `token.json` is saved afterward so you won't need to log in again.

## Usage

### Option A — Claude Desktop (conversational)

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "gmail-inbox-agent": {
      "command": "uv",
      "args": ["run", "--directory", "/full/path/to/gmail-inbox-agent", "gmail_mcp_server.py"]
    }
  }
}
```

Restart Claude Desktop, then ask things like *"check my inbox for anything important I might have missed"* or *"draft a reply to X apologizing for the delay."*

### Option B — Custom agent loop

```bash
uv run agent.py
```

Give it a task in plain English. It will show each tool call it decides to make, and pause for your approval before any write action (starring, archiving, creating a draft):

```
What should the agent do? Check my inbox for anything important I might have missed and star it

[Agent wants to call: find_important_unstarred({'max_results': 10})]
[Result: ...]
[Agent wants to call: star_email({'email_id': '...'})]
This is a write action. Approve? (y/n): y
[Result: Starred email ...]

Final answer:
I've starred 3 important emails I found...
```

## Design decisions worth noting

- **Scope minimization**: started with `gmail.readonly`, only widened to `gmail.modify` once a write action was actually needed
- **Never auto-send**: `create_draft` always saves to Drafts, never sends — the human reviews and sends manually
- **Human-in-the-loop on writes**: the custom agent loop (`agent.py`) explicitly pauses before any action that changes the inbox, distinct from read-only tools which run freely
- **Classification over keyword filters**: chose LLM-based judgment specifically because Gmail's native filters can't distinguish urgency by meaning, only by sender/keyword pattern

## Known limitations

- Gmail's native "Snooze" feature is not exposed by the public Gmail API, so it isn't implemented here
- Classification runs on a short content preview (`snippet`), not the full email body, to keep the implementation simple
- Single-user only — authorized against one Google account via personal OAuth, not designed for multi-tenant use

## What I'd build next

- Full email body parsing (multipart MIME) instead of relying on `snippet`
- A "needs a reply" tracker (emails received 2+ days ago with no response)
- Swap the hand-written agent loop for LangGraph to add persistent state and richer approval workflows
- Deploy as a remote MCP server (HTTP transport + OAuth) instead of local-only
