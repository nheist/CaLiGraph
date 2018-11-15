import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import util


def send_error(error_msg: str):
    if util.get_config('mailer.enable.error'):
        _send_mail('ERROR', error_msg)


def send_success(success_msg: str = 'Finished successfully.'):
    if util.get_config('mailer.enable.success'):
        _send_mail('SUCCESS', success_msg)


def _send_mail(subject, content):
    msg = MIMEMultipart()
    msg['From'] = util.get_config('mailer.sender')
    msg['To'] = util.get_config('mailer.receiver')
    msg['Subject'] = util.get_config('mailer.subject') + subject
    msg.attach(MIMEText(content, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(util.get_config('mailer.sender'), util.get_config('mailer.password'))
    server.send_message(msg)
    server.quit()
