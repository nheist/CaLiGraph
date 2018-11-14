import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import util


def send_error(error_msg: str):
    msg = MIMEMultipart()
    msg['From'] = util.get_config('mailer.sender')
    msg['To'] = util.get_config('mailer.receiver')
    msg['Subject'] = util.get_config('mailer.subject')
    msg.attach(MIMEText(error_msg, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(util.get_config('mailer.sender'), util.get_config('mail.password'))
    server.send_message(msg)
    server.quit()
