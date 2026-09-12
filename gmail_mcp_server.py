from mcp.server.fastmcp import FastMCP
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import os
import base64 as b64
from openai import OpenAI
from dotenv import load_dotenv
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

mcp = FastMCP("gmail-inbox-agent")


def get_gmail_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def get_email_date(headers):
    """Extract and parse the Date header from an email's headers list."""
    date_str = next((h["value"] for h in headers if h["name"] == "Date"), None)
    if not date_str:
        return None
    try:
        return parsedate_to_datetime(date_str)
    except Exception:
        return None


def is_today(email_date):
    """Check if a parsed email date falls on today's date (UTC-based)."""
    if email_date is None:
        return False
    now = datetime.now(timezone.utc)
    email_date_utc = email_date.astimezone(timezone.utc)
    return email_date_utc.date() == now.date()


def to_markdown_table(headers, rows):
    """Format a list of rows into a markdown table Claude renders nicely."""
    if not rows:
        return None
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        # Escape pipe characters so they don't break the table structure
        safe_row = [str(cell).replace("|", "/") for cell in row]
        lines.append("| " + " | ".join(safe_row) + " |")
    return "\n".join(lines)


def parse_classification(classification: str):
    """Split a classify_email result into (category, reason)."""
    category = "Unknown"
    reason = ""
    for line in classification.splitlines():
        if line.startswith("Category:"):
            category = line.replace("Category:", "").strip()
        elif line.startswith("Reason:"):
            reason = line.replace("Reason:", "").strip()
    return category, reason


@mcp.tool()
def list_recent_emails(max_results: int = 5) -> str:
    """
    List the most recent emails in the inbox, formatted as a markdown table.

    Parameters:
    max_results (int): How many recent emails to fetch.

    Returns:
    str: A markdown table of email ID, sender, and subject.
    """
    service = get_gmail_service()
    results = service.users().messages().list(userId="me", maxResults=max_results).execute()
    messages = results.get("messages", [])

    rows = []
    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        headers = msg_data["payload"]["headers"]
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(no subject)")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "(unknown sender)")
        rows.append((msg["id"], sender, subject))

    table = to_markdown_table(["ID", "From", "Subject"], rows)
    return table or "No recent emails found."


@mcp.tool()
def list_todays_emails(max_results: int = 20) -> str:
    """
    List emails received today only, formatted as a markdown table.

    Parameters:
    max_results (int): How many recent emails to scan (only today's are returned).

    Returns:
    str: A markdown table of today's emails' ID, time, sender, and subject.
    """
    service = get_gmail_service()
    results = service.users().messages().list(userId="me", maxResults=max_results).execute()
    messages = results.get("messages", [])

    rows = []
    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        headers = msg_data["payload"]["headers"]

        email_date = get_email_date(headers)
        if not is_today(email_date):
            continue

        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(no subject)")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "(unknown sender)")
        time_str = email_date.strftime("%H:%M") if email_date else "unknown time"
        rows.append((msg["id"], time_str, sender, subject))

    table = to_markdown_table(["ID", "Time", "From", "Subject"], rows)
    return table or "No emails received today."


def get_email_body(service, email_id):
    msg_data = service.users().messages().get(userId="me", id=email_id, format="full").execute()
    headers = msg_data["payload"]["headers"]
    subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(no subject)")
    sender = next((h["value"] for h in headers if h["name"] == "From"), "(unknown sender)")
    snippet = msg_data.get("snippet", "")
    return subject, sender, snippet


@mcp.tool()
def classify_email(email_id: str) -> str:
    """
    Classify an email's importance using the LLM.

    Parameters:
    email_id (str): The Gmail message ID to classify.

    Returns:
    str: One of Urgent, Important, Normal, Promotional, Social, Updates,
    Forums, or Purchases, with a one-line reason.
    """
    service = get_gmail_service()
    subject, sender, snippet = get_email_body(service, email_id)

    prompt = f"""Classify this email into exactly one category:
Urgent, Important, Normal, Promotional, Social, Updates, Forums, or Purchases.

Use these definitions:
- Urgent: needs action very soon (deadline, emergency, time-sensitive request)
- Important: matters but not time-critical (job/interview, college notice, personal)
- Normal: routine email, nothing pressing
- Promotional: marketing, sales, ads, discounts
- Social: notifications from social networks (LinkedIn, Instagram, etc.)
- Updates: automated notifications (bills, receipts, confirmations, alerts)
- Forums: mailing lists, newsletters, group discussions
- Purchases: order confirmations, shipping updates, delivery notices

From: {sender}
Subject: {subject}
Preview: {snippet}

Respond in this exact format:
Category: <category>
Reason: <one short sentence>"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        max_tokens=300,
        reasoning_effort="low",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


@mcp.tool()
def star_email(email_id: str) -> str:
    """
    Star an email in Gmail.

    Parameters:
    email_id (str): The Gmail message ID to star.

    Returns:
    str: Confirmation message.
    """
    service = get_gmail_service()
    service.users().messages().modify(
        userId="me",
        id=email_id,
        body={"addLabelIds": ["STARRED"]},
    ).execute()
    return f"Starred email {email_id}"


@mcp.tool()
def find_important_unstarred(max_results: int = 10, today_only: bool = False) -> str:
    """
    Scan recent emails, classify each one, and return the ones judged
    Urgent or Important that are not already starred, as a markdown table.

    Parameters:
    max_results (int): How many recent emails to scan.
    today_only (bool): If True, only consider emails received today.

    Returns:
    str: A markdown table of matching emails with date, subject, category, and reason.
    """
    service = get_gmail_service()
    results = service.users().messages().list(userId="me", maxResults=max_results).execute()
    messages = results.get("messages", [])

    rows = []
    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        label_ids = msg_data.get("labelIds", [])

        if "STARRED" in label_ids:
            continue  # already starred, skip it

        headers = msg_data["payload"]["headers"]
        email_date = get_email_date(headers)

        if today_only and not is_today(email_date):
            continue

        classification = classify_email(msg["id"])
        category, reason = parse_classification(classification)

        if category in ("Urgent", "Important"):
            subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(no subject)")
            date_str = email_date.strftime("%Y-%m-%d %H:%M") if email_date else "unknown date"
            rows.append((msg["id"], date_str, subject, category, reason))

    table = to_markdown_table(["ID", "Date", "Subject", "Category", "Reason"], rows)
    if table:
        return table
    scope = "today" if today_only else "recent emails"
    return f"No important unstarred emails found in {scope}."


def build_raw_message(to: str, subject: str, body: str) -> str:
    """Shared helper: build a base64url-encoded MIME message for Gmail's API."""
    from email.mime.text import MIMEText

    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject
    return b64.urlsafe_b64encode(message.as_bytes()).decode()


def get_draft_content(service, draft_id):
    """Fetch a draft's to/subject/body as plain values."""
    draft = service.users().drafts().get(userId="me", id=draft_id, format="full").execute()
    msg = draft["message"]
    headers = msg["payload"]["headers"]

    to = next((h["value"] for h in headers if h["name"] == "To"), "")
    subject = next((h["value"] for h in headers if h["name"] == "Subject"), "")

    body_data = msg["payload"].get("body", {}).get("data", "")
    if not body_data and "parts" in msg["payload"]:
        body_data = msg["payload"]["parts"][0]["body"].get("data", "")

    body_text = b64.urlsafe_b64decode(body_data).decode("utf-8") if body_data else ""
    return to, subject, body_text


@mcp.tool()
def create_draft(to: str, subject: str, rough_text: str) -> str:
    """
    Clean up rough, informal text into a proper email and save it as a
    Gmail draft. Never sends automatically — the user must review and
    send it manually from Gmail (or use send_draft after reviewing).

    Parameters:
    to (str): Recipient's email address.
    subject (str): Email subject line.
    rough_text (str): The user's rough draft, may contain typos or
    informal grammar.

    Returns:
    str: Confirmation message including the new draft's ID.
    """
    prompt = f"""Rewrite this into a polite, professional, grammatically
correct email. Fix any typos or grammar issues. Keep the original intent
and tone level (don't make it overly formal if it wasn't meant to be).
Return ONLY the email body text, nothing else — no subject line, no
explanation.

Rough draft: {rough_text}"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    clean_body = response.choices[0].message.content

    service = get_gmail_service()
    raw_message = build_raw_message(to, subject, clean_body)

    draft = service.users().drafts().create(
        userId="me",
        body={"message": {"raw": raw_message}},
    ).execute()

    return f"Draft created (ID: {draft['id']}). Review it in your Gmail Drafts folder before sending."


@mcp.tool()
def update_draft(draft_id: str, to: str, subject: str, rough_text: str) -> str:
    """
    Replace an existing draft entirely with new content — use this when
    you want to rewrite the draft from scratch. Runs the new text through
    the same grammar/tone cleanup as create_draft. To make a small change
    to existing text instead (e.g. "remove this line"), use edit_draft.

    Parameters:
    draft_id (str): The Gmail draft ID to update.
    to (str): New recipient's email address.
    subject (str): New email subject line.
    rough_text (str): New rough text to clean up and use as the body.

    Returns:
    str: Confirmation message.
    """
    prompt = f"""Rewrite this into a polite, professional, grammatically
correct email. Fix any typos or grammar issues. Keep the original intent
and tone level (don't make it overly formal if it wasn't meant to be).
Return ONLY the email body text, nothing else — no subject line, no
explanation.

Rough draft: {rough_text}"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    clean_body = response.choices[0].message.content

    service = get_gmail_service()
    raw_message = build_raw_message(to, subject, clean_body)

    service.users().drafts().update(
        userId="me",
        id=draft_id,
        body={"message": {"raw": raw_message}},
    ).execute()

    return f"Draft {draft_id} updated. Review it in your Gmail Drafts folder before sending."


@mcp.tool()
def edit_draft(draft_id: str, instruction: str) -> str:
    """
    Make a targeted edit to an EXISTING draft's body, based on the
    current content plus a plain-English instruction (e.g. "remove the
    line about not attending the test" or "make it more formal").
    Keeps the same recipient and subject unless the instruction says
    otherwise. Still saved as a draft only, never sent.

    Parameters:
    draft_id (str): The Gmail draft ID to edit.
    instruction (str): Plain-English description of the change to make.

    Returns:
    str: Confirmation message, including the updated body for review.
    """
    service = get_gmail_service()
    to, subject, current_body = get_draft_content(service, draft_id)

    prompt = f"""Here is the current email body:

---
{current_body}
---

Apply this edit: {instruction}

Return ONLY the full updated email body, nothing else — no subject
line, no explanation, no markers like "---"."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    updated_body = response.choices[0].message.content

    raw_message = build_raw_message(to, subject, updated_body)
    service.users().drafts().update(
        userId="me",
        id=draft_id,
        body={"message": {"raw": raw_message}},
    ).execute()

    return f"Draft {draft_id} edited.\n\nUpdated content:\nTo: {to}\nSubject: {subject}\n\n{updated_body}"


@mcp.tool()
def delete_draft(draft_id: str) -> str:
    """
    Permanently delete a draft. This does not affect any sent email —
    only removes the draft itself.

    Parameters:
    draft_id (str): The Gmail draft ID to delete.

    Returns:
    str: Confirmation message.
    """
    service = get_gmail_service()
    service.users().drafts().delete(userId="me", id=draft_id).execute()
    return f"Draft {draft_id} deleted."


@mcp.tool()
def send_draft(draft_id: str) -> str:
    """
    Send an existing draft as-is. THIS IS IRREVERSIBLE — the email will
    actually be delivered to the recipient. Always confirm the draft's
    content with get_draft before calling this. Once sent, it will also
    appear in list_sent_emails (Gmail applies this automatically).

    Parameters:
    draft_id (str): The Gmail draft ID to send.

    Returns:
    str: Confirmation message including the sent message's ID.
    """
    service = get_gmail_service()
    result = service.users().drafts().send(
        userId="me",
        body={"id": draft_id},
    ).execute()
    return f"Draft {draft_id} sent successfully (message ID: {result['id']})."


@mcp.tool()
def list_sent_emails(max_results: int = 10) -> str:
    """
    List emails you have sent to others, formatted as a markdown table.
    This includes emails sent via send_draft, since Gmail automatically
    labels sent drafts the same as any other sent email.

    Parameters:
    max_results (int): How many sent emails to fetch.

    Returns:
    str: A markdown table of sent emails' ID, recipient, and subject.
    """
    service = get_gmail_service()
    results = service.users().messages().list(
        userId="me", maxResults=max_results, labelIds=["SENT"]
    ).execute()
    messages = results.get("messages", [])

    rows = []
    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        headers = msg_data["payload"]["headers"]
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(no subject)")
        recipient = next((h["value"] for h in headers if h["name"] == "To"), "(unknown recipient)")
        rows.append((msg["id"], recipient, subject))

    table = to_markdown_table(["ID", "To", "Subject"], rows)
    return table or "No sent emails found."


@mcp.tool()
def archive_email(email_id: str) -> str:
    """
    Archive an email (remove it from Inbox, keep it searchable).

    Parameters:
    email_id (str): The Gmail message ID to archive.

    Returns:
    str: Confirmation message.
    """
    service = get_gmail_service()
    service.users().messages().modify(
        userId="me",
        id=email_id,
        body={"removeLabelIds": ["INBOX"]},
    ).execute()
    return f"Archived email {email_id}"


@mcp.tool()
def list_archived_emails(max_results: int = 10) -> str:
    """
    List emails that have been archived, formatted as a markdown table.

    Parameters:
    max_results (int): How many archived emails to fetch.

    Returns:
    str: A markdown table of archived emails' ID, sender, and subject.
    """
    service = get_gmail_service()
    results = service.users().messages().list(
        userId="me", maxResults=max_results, q="-in:inbox -in:trash"
    ).execute()
    messages = results.get("messages", [])

    rows = []
    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        headers = msg_data["payload"]["headers"]
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(no subject)")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "(unknown sender)")
        rows.append((msg["id"], sender, subject))

    table = to_markdown_table(["ID", "From", "Subject"], rows)
    return table or "No archived emails found."


@mcp.tool()
def list_spam_emails(max_results: int = 10) -> str:
    """
    List emails currently in Spam, formatted as a markdown table.

    Parameters:
    max_results (int): How many spam emails to fetch.

    Returns:
    str: A markdown table of spam emails' ID, sender, and subject.
    """
    service = get_gmail_service()
    results = service.users().messages().list(
        userId="me", maxResults=max_results, labelIds=["SPAM"]
    ).execute()
    messages = results.get("messages", [])

    rows = []
    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        headers = msg_data["payload"]["headers"]
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(no subject)")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "(unknown sender)")
        rows.append((msg["id"], sender, subject))

    table = to_markdown_table(["ID", "From", "Subject"], rows)
    return table or "No spam emails found."


@mcp.tool()
def list_trash_emails(max_results: int = 10) -> str:
    """
    List emails currently in Trash, formatted as a markdown table.

    Parameters:
    max_results (int): How many trashed emails to fetch.

    Returns:
    str: A markdown table of trashed emails' ID, sender, and subject.
    """
    service = get_gmail_service()
    results = service.users().messages().list(
        userId="me", maxResults=max_results, labelIds=["TRASH"]
    ).execute()
    messages = results.get("messages", [])

    rows = []
    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        headers = msg_data["payload"]["headers"]
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(no subject)")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "(unknown sender)")
        rows.append((msg["id"], sender, subject))

    table = to_markdown_table(["ID", "From", "Subject"], rows)
    return table or "No trashed emails found."


@mcp.tool()
def list_starred_emails(max_results: int = 10) -> str:
    """
    List emails currently starred in Gmail, formatted as a markdown table.

    Parameters:
    max_results (int): How many starred emails to fetch.

    Returns:
    str: A markdown table of starred emails' ID, sender, and subject.
    """
    service = get_gmail_service()
    results = service.users().messages().list(
        userId="me", maxResults=max_results, labelIds=["STARRED"]
    ).execute()
    messages = results.get("messages", [])

    rows = []
    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        headers = msg_data["payload"]["headers"]
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(no subject)")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "(unknown sender)")
        rows.append((msg["id"], sender, subject))

    table = to_markdown_table(["ID", "From", "Subject"], rows)
    return table or "No starred emails found."


@mcp.tool()
def get_draft(draft_id: str) -> str:
    """
    Get the full content of a specific draft, including the actual email text.

    Parameters:
    draft_id (str): The Gmail draft ID to fetch.

    Returns:
    str: The recipient, subject, and full body text of the draft.
    """
    service = get_gmail_service()
    to, subject, body_text = get_draft_content(service, draft_id)
    if not body_text:
        body_text = "(empty body)"
    return f"To: {to}\nSubject: {subject}\n\n{body_text}"


@mcp.tool()
def list_drafts(max_results: int = 10) -> str:
    """
    List all current drafts, formatted as a markdown table.

    Parameters:
    max_results (int): How many drafts to fetch.

    Returns:
    str: A markdown table of draft ID, recipient, and subject.
    """
    service = get_gmail_service()
    results = service.users().drafts().list(userId="me", maxResults=max_results).execute()
    drafts = results.get("drafts", [])

    rows = []
    for d in drafts:
        draft = service.users().drafts().get(userId="me", id=d["id"]).execute()
        headers = draft["message"]["payload"]["headers"]
        to = next((h["value"] for h in headers if h["name"] == "To"), "(no recipient)")
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(no subject)")
        rows.append((d["id"], to, subject))

    table = to_markdown_table(["Draft ID", "To", "Subject"], rows)
    return table or "No drafts found."


if __name__ == "__main__":
    mcp.run()