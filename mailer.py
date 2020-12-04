"""Mailer to inform about success or failure of the extraction."""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import utils


def send_error(error_msg: str):
    if utils.get_config('mailer.enable.error'):
        _send_mail('ERROR', error_msg)


def send_success(success_msg: str = 'Finished successfully.'):
    if utils.get_config('mailer.enable.success'):
        _send_mail('SUCCESS', success_msg)


def _send_mail(subject, content):
    msg = MIMEMultipart()
    msg['From'] = utils.get_config('mailer.sender')
    msg['To'] = utils.get_config('mailer.receiver')
    msg['Subject'] = utils.get_config('mailer.subject') + subject
    msg.attach(MIMEText(content, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(utils.get_config('mailer.sender'), utils.get_config('mailer.password'))
    server.send_message(msg)
    server.quit()
