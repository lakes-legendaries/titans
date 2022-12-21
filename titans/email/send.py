"""Email sender"""
from __future__ import annotations

from base64 import b64encode
from copy import deepcopy
import json
from os.path import basename
from typing import Union

import requests

from titans.sql import connect


class SendEmails:
    """Send emails

    Parameters
    ----------
    subject: str
        Email subject
    body: str
        Email body. If this is a filename ending in :code:`.html`, then this
        will be read in and treated like a file. Otherwise, the text contained
        here will be used as the message body.
    attachments: list[str], optional, default=[]
        Attachment filenames
    recipients: str or list[str], optional, default='mike@lakeslegendaries.com'
        specify receipient emails. Ignored if :code:`send_all`
    send_to_all: bool, optional, default=False
        Send to all recipients on email list
    """

    def __init__(
        self,
        /,
        subject: str,
        body: str,
        *,
        attachments: list[str] = [],
        recipients: Union[str, list[str]] = 'mike@lakeslegendaries.com',
        send_to_all: bool = False,
    ):

        # save passed
        self._subject = subject
        self._body = body
        self._attachments = attachments
        self._recipients = recipients
        self._send_to_all = send_to_all

        # create email dictionary
        email_dict = self._create_dict()

        # get recepients
        recipients = self._get_recipients()

        # send
        self._send(email_dict, recipients)

    def _create_dict(self) -> dict:
        """Create email dictionary

        Returns
        -------
        dict
            email dictionary
        """

        # read email body from html file
        is_html = self._body.endswith('.html')
        body = (
            self._body
            if not is_html
            else open(self._body, 'r').read()
        )

        # create email dictionary
        email_dict = {
            'message': {
                'subject': self._subject,
                'body': {
                    'contentType': 'HTML' if is_html else 'Text',
                    'content': body,
                },
                'attachments': [],
            },
            'saveToSentItems': False,
        }

        # add attachments
        for attachment in self._attachments:
            bytes = b64encode(open(attachment, 'rb').read()).decode('utf-8')
            email_dict['message']['attachments'].append({
                '@odata.type': '#microsoft.graph.fileAttachment',
                'name': basename(attachment),
                'isInline': True,
                'contentId': basename(attachment).rsplit('.')[0],
                'contentType': f"image/{basename(attachment).rsplit('.')[1]}",
                'contentBytes': bytes,
            })

        # return
        return email_dict

    def _get_recipients(self) -> list[str]:
        """Get recipient list

        Returns
        -------
        list[str]
            recipients
        """

        # send to all recepients
        if self._send_to_all:
            emails = connect().execute("""SELECT email FROM Contacts""")
            recipients = [
                email[0]
                for email in emails
            ]

        # send to provided recipients
        else:

            # load repients
            recipients = self._recipients

            # harmonize parameter types
            if type(recipients) is str:
                recipients = [recipients]

        # get confirmation for large sends
        if len(recipients) > 5:
            prompt = (
                f'Sending to {len(recipients)} contacts. '
                'Type \'Y\' to continue: '
            )
            if input(prompt) != 'Y':
                print('Aborting')
                quit()

        # return
        return recipients

    def _send(self, /, email_dict: dict, recipients: list[str]):
        """Send emails

        Parameters
        ----------
        email_dict: dict
            email to send
        recipients: list[str]
            email recipients
        """

        # send to each recipient
        for recipient in recipients:

            # get fresh copy of email
            email = deepcopy(email_dict)

            # substitute in email address
            email['message']['body']['content'] = \
                email['message']['body']['content'].replace(
                    r'#|EMAIL|#',
                    recipient,
                )

            # add recipient to email dictionary
            email['message']['toRecipients'] = [
                {
                    'emailAddress': {
                        'address': recipient,
                    },
                },
            ]

            # send email
            try:
                # connect to database
                engine = connect()

                # lock credentials table
                engine.execute('LOCK TABLES creds WRITE')

                # pull credentials
                creds = {
                    key: value
                    for key, value in engine.execute(
                        'SELECT * FROM creds'
                    ).fetchall()
                }

                # refresh email token
                url = 'https://login.microsoftonline.com'
                refreshed = requests.get(
                    f"{url}/{creds['tenant']}/oauth2/v2.0/token",
                    data={
                        'client_id': creds['client_id'],
                        'client_secret': creds['client_secret'],
                        'scope': 'offline_access mail.send',
                        'refresh_token': creds['refresh_token'],
                        'grant_type': 'refresh_token',
                    },
                    headers={
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                ).json()

                # upload refreshed token
                for field in ['refresh_token', 'access_token']:
                    engine.execute(f"""
                        UPDATE creds
                        SET Value = "{refreshed[field]}"
                        WHERE Name = "{field}"
                    """)

                # send email
                response = requests.post(
                    'https://graph.microsoft.com/v1.0/me/sendMail',
                    data=json.dumps(email),
                    headers={
                        'Authorization': refreshed['access_token'],
                        'Content-type': 'application/json',
                        'Host': 'graph.microsoft.com',
                    },
                )

                # make sure completed successfully
                if response.status_code == 400:
                    print(f'Failed to send to {recipient}')

            # clean-up
            finally:

                # unlock email credentials table
                engine.execute('UNLOCK TABLES')
