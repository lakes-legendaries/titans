#!/bin/bash

# exit on error
set -e

# load credentials
. $SECRETS_DIR/titans-email-creds

# build URL
URL=$(echo \
    "https://login.microsoftonline.com/$TENANT/oauth2/v2.0/authorize" \
    "?client_id=$CLIENT_ID" \
    "&response_type=code" \
    "&response_mode=query" \
    "&scope=offline_access%20mail.send" \
)

# direct user to url
echo "Authenaticate at: $(echo $URL | sed 's/ //g')"
