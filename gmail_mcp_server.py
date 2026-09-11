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


@mcp.tool()
def list_recent_emails(max_results: int = 5) -> str:
    """
    List the most recent emails in the inbox.

    Parameters:
    max_results (int): How many recent emails to fetch.

    Returns:
    str: A formatted list of email ID, sender, and subject for each email.
    """
    service = get_gmail_service()
    results = service.users().messages().list(userId="me", maxResults=max_results).execute()
    messages = results.get("messages", [])

    lines = []
    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        headers = msg_data["payload"]["headers"]
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(no subject)")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "(unknown sender)")
        lines.append(f"ID: {msg['id']} | From: {sender} | Subject: {subject}")

    if not lines:
        return "No recent emails found."
    return "\n".join(lines)


@mcp.tool()
def list_todays_emails(max_results: int = 20) -> str:
    """
    List emails received today only.

    Parameters:
    max_results (int): How many recent emails to scan (only today's are returned).

    Returns:
    str: A formatted list of today's emails' ID, time, sender, and subject.
    """
    service = get_gmail_service()
    results = service.users().messages().list(userId="me", maxResults=max_results).execute()
    messages = results.get("messages", [])

    lines = []
    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        headers = msg_data["payload"]["headers"]

        email_date = get_email_date(headers)
        if not is_today(email_date):
            continue

        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(no subject)")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "(unknown sender)")
        time_str = email_date.strftime("%H:%M") if email_date else "unknown time"
        lines.append(f"ID: {msg['id']} | Time: {time_str} | From: {sender} | Subject: {subject}")

    if not lines:
        return "No emails received today."
    return "\n".join(lines)


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
    Urgent or Important that are not already starred. Each result
    includes the date/time the email was received.

    Parameters:
    max_results (int): How many recent emails to scan.
    today_only (bool): If True, only consider emails received today.

    Returns:
    str: A list of matching emails with their date, subject, and reason.
    """
    service = get_gmail_service()
    results = service.users().messages().list(userId="me", maxResults=max_results).execute()
    messages = results.get("messages", [])

    flagged = []
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

        if "Category: Urgent" in classification or "Category: Important" in classification:
            subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(no subject)")
            date_str = email_date.strftime("%Y-%m-%d %H:%M") if email_date else "unknown date"
            flagged.append(f"ID: {msg['id']} | Date: {date_str} | Subject: {subject} | {classification}")

    if not flagged:
        scope = "today" if today_only else "recent emails"
        return f"No important unstarred emails found in {scope}."
    return "\n".join(flagged)


def build_raw_message(to: str, subject: str, body: str) -> str:
    """Shared helper: build a base64url-encoded MIME message for Gmail's API."""
    from email.mime.text import MIMEText

    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject
    return b64.urlsafe_b64encode(message.as_bytes()).decode()


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
    Edit an existing draft — replaces its recipient, subject, and body.
    Runs the new text through the same grammar/tone cleanup as
    create_draft. Still saved as a draft only, never sent.

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
def send_draft(draft_id: str) -> str:
    """
    Send an existing draft as-is. THIS IS IRREVERSIBLE — the email will
    actually be delivered to the recipient. Always confirm the draft's
    content with get_draft before calling this.

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
    List emails you have sent to others.

    Parameters:
    max_results (int): How many sent emails to fetch.

    Returns:
    str: A formatted list of sent emails' ID, recipient, and subject.
    """
    service = get_gmail_service()
    results = service.users().messages().list(
        userId="me", maxResults=max_results, labelIds=["SENT"]
    ).execute()
    messages = results.get("messages", [])

    lines = []
    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        headers = msg_data["payload"]["headers"]
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(no subject)")
        recipient = next((h["value"] for h in headers if h["name"] == "To"), "(unknown recipient)")
        lines.append(f"ID: {msg['id']} | To: {recipient} | Subject: {subject}")

    if not lines:
        return "No sent emails found."
    return "\n".join(lines)


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
    List emails that have been archived (not in Inbox, not Trash).

    Parameters:
    max_results (int): How many archived emails to fetch.

    Returns:
    str: A formatted list of archived emails' ID, sender, and subject.
    """
    service = get_gmail_service()
    results = service.users().messages().list(
        userId="me", maxResults=max_results, q="-in:inbox -in:trash"
    ).execute()
    messages = results.get("messages", [])

    lines = []
    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        headers = msg_data["payload"]["headers"]
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(no subject)")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "(unknown sender)")
        lines.append(f"ID: {msg['id']} | From: {sender} | Subject: {subject}")

    if not lines:
        return "No archived emails found."
    return "\n".join(lines)


@mcp.tool()
def list_spam_emails(max_results: int = 10) -> str:
    """
    List emails currently in Spam.

    Parameters:
    max_results (int): How many spam emails to fetch.

    Returns:
    str: A formatted list of spam emails' ID, sender, and subject.
    """
    service = get_gmail_service()
    results = service.users().messages().list(
        userId="me", maxResults=max_results, labelIds=["SPAM"]
    ).execute()
    messages = results.get("messages", [])

    lines = []
    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        headers = msg_data["payload"]["headers"]
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(no subject)")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "(unknown sender)")
        lines.append(f"ID: {msg['id']} | From: {sender} | Subject: {subject}")

    if not lines:
        return "No spam emails found."
    return "\n".join(lines)


@mcp.tool()
def list_trash_emails(max_results: int = 10) -> str:
    """
    List emails currently in Trash.

    Parameters:
    max_results (int): How many trashed emails to fetch.

    Returns:
    str: A formatted list of trashed emails' ID, sender, and subject.
    """
    service = get_gmail_service()
    results = service.users().messages().list(
        userId="me", maxResults=max_results, labelIds=["TRASH"]
    ).execute()
    messages = results.get("messages", [])

    lines = []
    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        headers = msg_data["payload"]["headers"]
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(no subject)")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "(unknown sender)")
        lines.append(f"ID: {msg['id']} | From: {sender} | Subject: {subject}")

    if not lines:
        return "No trashed emails found."
    return "\n".join(lines)


@mcp.tool()
def list_starred_emails(max_results: int = 10) -> str:
    """
    List emails currently starred in Gmail.

    Parameters:
    max_results (int): How many starred emails to fetch.

    Returns:
    str: A formatted list of starred emails' ID, sender, and subject.
    """
    service = get_gmail_service()
    results = service.users().messages().list(
        userId="me", maxResults=max_results, labelIds=["STARRED"]
    ).execute()
    messages = results.get("messages", [])

    lines = []
    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        headers = msg_data["payload"]["headers"]
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(no subject)")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "(unknown sender)")
        lines.append(f"ID: {msg['id']} | From: {sender} | Subject: {subject}")

    if not lines:
        return "No starred emails found."
    return "\n".join(lines)


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
    draft = service.users().drafts().get(userId="me", id=draft_id, format="full").execute()
    msg = draft["message"]
    headers = msg["payload"]["headers"]

    to = next((h["value"] for h in headers if h["name"] == "To"), "(no recipient)")
    subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(no subject)")

    body_data = msg["payload"].get("body", {}).get("data", "")
    if not body_data and "parts" in msg["payload"]:
        body_data = msg["payload"]["parts"][0]["body"].get("data", "")

    body_text = b64.urlsafe_b64decode(body_data).decode("utf-8") if body_data else "(empty body)"

    return f"To: {to}\nSubject: {subject}\n\n{body_text}"


@mcp.tool()
def list_drafts(max_results: int = 10) -> str:
    """
    List all current drafts with their IDs and subjects.

    Parameters:
    max_results (int): How many drafts to fetch.

    Returns:
    str: A formatted list of draft ID, recipient, and subject.
    """
    service = get_gmail_service()
    results = service.users().drafts().list(userId="me", maxResults=max_results).execute()
    drafts = results.get("drafts", [])

    lines = []
    for d in drafts:
        draft = service.users().drafts().get(userId="me", id=d["id"]).execute()
        headers = draft["message"]["payload"]["headers"]
        to = next((h["value"] for h in headers if h["name"] == "To"), "(no recipient)")
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(no subject)")
        lines.append(f"Draft ID: {d['id']} | To: {to} | Subject: {subject}")

    if not lines:
        return "No drafts found."
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()