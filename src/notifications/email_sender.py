import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SENDER_EMAIL = "sparshj2003@gmail.com"
APP_PASSWORD = "gvai xykc roms wvgr"


def send_uniform_violation_email(student_roll, student_email, violation_time, fine_amount):
    try:
        # Email setup
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
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

        # Connect to Gmail SMTP
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.sendmail(SENDER_EMAIL, student_email, msg.as_string())

        print(f"✔ Email successfully sent to {student_email}")

    except Exception as e:
        print(f"❌ Error sending email: {e}")
