"""FastAPI app"""

import re
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yaml

from titans.email import SendEmails

from titans import __version__
from titans.sql import connect, sanitize


# create app
app = FastAPI()

# bundle app info
app_info = {
    'service': 'titans.api',
    'version': __version__,
}

# allow cors access
app.add_middleware(
    CORSMiddleware,
    allow_headers=["*"],
    allow_methods=["*"],
    allow_origins=["*"],
)


@app.get('/')
def home():
    """Root prompt"""
    return app_info


@app.get('/subscribe/{email}')
def subscribe(email: str):
    """Subscribe to updates

    Parameters
    ----------
    email: str
        email to subscribe
    """

    # sanitize input
    email = sanitize(email).lower()

    # get job info
    job_info = {
        **app_info,
        'operation': 'subscribe',
        'email': email,
    }

    # check conforms to expected format
    if re.fullmatch(r'[^@]+@[^@.]+\.[^@.]+', email) is None:
        return {
            **app_info,
            'success': False,
            'reason': 'Invalid email address format',
        }

    # check if email is already in table
    engine = connect()
    count = engine.execute(f"""
        SELECT COUNT(*) FROM contacts
        WHERE Email = "{email}"
    """).fetchone()[0]
    if count:
        return {
            **job_info,
            'success': False,
            'reason': 'Email address is already subscribed',
        }

    # insert into table
    engine.execute(f"""
        INSERT INTO contacts (Email)
        VALUES ("{email}")
    """)

    # send welcome email
    try:
        email_config = yaml.safe_load(open("email/test/config.yaml"))
        SendEmails(**email_config)
    except Exception:
        return {
            **job_info,
            'success': False,
            'reason': 'Failed to send welcome email',
        }

    # return success
    return {
        **job_info,
        'success': True,
    }


@app.get('/unsubscribe/{email}')
def unsubscribe(email: str):
    """Unsubscribe from updates

    Parameters
    ----------
    email: str
        email to unsubscribe
    """

    # sanitize input
    email = sanitize(email).lower()

    # get job info
    job_info = {
        **app_info,
        'operation': 'unsubscribe',
        'email': email,
    }

    # check if email is subscribed
    engine = connect()
    count = engine.execute(f"""
        SELECT COUNT(*) FROM contacts
        WHERE Email = "{email}"
    """).fetchone()[0]
    if not count:
        return {
            **job_info,
            'success': False,
            'reason': "Email address doesn't exist in database",
        }

    # insert into table
    engine.execute(f"""
        DELETE FROM contacts
        WHERE Email = "{email}"
    """)
    return {
        **job_info,
        'success': True,
    }


@app.get('/comment/')
def comment(comment: str, email: Optional[str] = ''):
    """Leave a question or comment

    Parameters
    ----------
    comment: str
        comment or question
    email: str, optional, default=''
        commenter's email address
    """

    # sanitize input
    comment = sanitize(comment)
    email = sanitize(email).lower()

    # get job info
    job_info = {
        **app_info,
        'operation': 'comment',
        'comment': comment,
        'email': email,
    }

    # insert into table
    connect().execute(f"""
        INSERT INTO comments (Comment, Email)
        VALUES ("{comment}", "{email}")
    """)

    # email comment
    try:
        SendEmails(
            subject='New Question / Comment',
            body=comment if not email else f'{comment}\n\n- {email}',
        )
    except Exception:
        return {
            **job_info,
            'success': False,
            'reason': 'Failed to email comments',
        }

    # return success
    return {
        **job_info,
        'success': True,
    }


@app.get('/poll/{name}')
def poll(name: str, email: str, response: str):
    """Respond to a poll

    Parameters
    ----------
    name: str
        name of poll
    email: str
        email of responder
    response: str
        response to poll
    """

    # sanitize input
    name = sanitize(name)
    email = sanitize(email).lower()
    response = sanitize(response)

    # get job info
    job_info = {
        **app_info,
        'operation': 'poll',
        'name': name,
        'email': email,
        'response': response,
    }

    # check if email already responded
    engine = connect()
    count = engine.execute(f"""
        SELECT COUNT(*) FROM {name}
        WHERE Email = "{email}"
    """).fetchone()[0]
    if count:
        return {
            **job_info,
            'success': False,
            'reason': "Email address has already responded",
        }

    # insert response into table
    connect().execute(f"""
        INSERT INTO {name} (Email, Response)
        VALUES ("{email}", "{response}")
    """)

    # return success
    return {
        **job_info,
        'success': True,
    }
