# src/notifications/email_sender.py
"""Email notifications for uniform violations — credentials loaded from environment."""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from src.config import Config

logger = logging.getLogger(__name__)


def send_uniform_violation_email(student_roll, student_email, violation_time, fine_amount):
    """Send a uniform violation notification email to a student."""
    if not Config.SMTP_EMAIL or not Config.SMTP_PASSWORD:
        logger.warning("SMTP credentials not configured — skipping email for %s", student_roll)
        return False

    try:
        msg = MIMEMultipart()
        msg["From"] = Config.SMTP_EMAIL
        msg["To"] = student_email
        msg["Subject"] = "Uniform Violation Notice"

        body = f"""
        Dear Student ({student_roll}),

        You have been reported for a uniform violation.

        📅 Violation Time: {violation_time}
        💰 Fine Amount: ₹{fine_amount}

        Please resolve this at the earliest.

        Regards,
        Discipline Committee
        """

        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(Config.SMTP_HOST, Config.SMTP_PORT) as server:
            server.starttls()
            server.login(Config.SMTP_EMAIL, Config.SMTP_PASSWORD)
            server.sendmail(Config.SMTP_EMAIL, student_email, msg.as_string())

        logger.info("Email sent to %s (%s)", student_roll, student_email)
        return True

    except Exception as e:
        logger.error("Error sending email to %s: %s", student_email, e)
        return False
