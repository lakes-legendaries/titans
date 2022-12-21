#!/bin/bash

# exit on error
set -e

# load credentials
. $SECRETS_DIR/titans-email-creds

# create curl data body
DATA=$(echo \
    "client_id=$CLIENT_ID" \
    "&client_secret=$CLIENT_SECRET" \
    "&scope=offline_access%20mail.send" \
    "&grant_type=authorization_code" \
    "&code=$(cat $SECRETS_DIR/titans-email-code)" \
)

# fetch token
curl -sH "Content-Type: application/x-www-form-urlencoded" \
    -d "$DATA" https://login.microsoftonline.com/$TENANT/oauth2/v2.0/token \
> $SECRETS_DIR/titans-email-token
