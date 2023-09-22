#########
Cloud Ops
#########

********
Overview
********

This repo controls all of the cloud operations for Titans Of Eden. This
includes:

#. Website HTML/CSS/JS
#. Online Demo
#. Blender Video-Rendering & Formatting
#. Database Creation & Management
#. Web-based API
#. Email Listserv Management

***********
Cloud Infra
***********

The following infrastructure was setup for this project:

#. `(Production) File Server / Static Website
   <https://titansfileserver.z13.web.core.windows.net/>`_
#. `Dev File Server / Static Website
   <https://titansfileserverdev.z13.web.core.windows.net/>`_
#. Batch Account (for parallel computing / video rendering)
#. MySQL Server (as a general database)
#. Container Registry
#. `CDN (for public-facing website)
   <https://www.titansofeden.com>`_
#. `Linux VM <https://titansapi.eastus.cloudapp.azure.com/>`_ (which provides
   all API functionality)

All infrastructure resides on Azure.

********
Cloud VM
********

This project has a cloud VM. All development work should be carried out on this
VM (e.g. by SSH'ing into it).

The VM's IP is whitelisted, which allows it to access the database and other
cloud services. Additionally, credentials are securely stored on this VM.

If the cloud VM ever goes down and needs to be re-provisioned, once the VM is
up again, you can re-provision by:

#. Placing a MySQL connection string in :code:`~/secrets/titans-mysql`

#. Whitelisting the VM's IP

#. Running the command:

   .. code-block:: bash

      curl https://raw.githubusercontent.com/lakes-legendaries/titans/main/webserver/provision.sh | bash

This will install all dependencies, establish credentials for a secure HTTPS
connection, launch the API, and schedule regular updates and upgrades.

*******
Website
*******

All (most?) website code is contained in :code:`titans/website`.

Please note that I am **not** a frontend developer: Cut me some slack if this
doesn't look perfect!

HTML
====

HTML code is contained in the subfolder `html`. I wrote my own compiler for
HTML code: Comments that include the name of a file substitute in that file's
contents on compilation. E.g. the comment :code:`<!-- html/partial/include.html
-->` in :code:`html/constructed.html` substitutes in the contents of
`html/partial/include.html` on compilation.

Instructions on how to run the compiler are below.

CSS
===

All CSS code is in the subfolder :code:`style`, including code for dynamic
scaling to mobile / desktop screens.

JS
==

All JS code is located in the subfolder :code:`script`. JQuery was used
heavily. This code could definitely use some TLC by a real frontend developer
(e.g. within the React framework).

Compiling
=========

To run the compiler, simpy run:

.. code-block:: bash

   python titans/website/compile.py

and all html files will be compiled and output to :code:`titans/website/site`.

Assets
======

There are a handful of image files that are NOT managed by this repo, but are
expected to be uploaded to the dev fileserver by other processes. This includes
various card images, instructional images, and rules PDFs, among some other
scattered files. The code to upload those files lives with each of the
respective files.

(Some other assets, like videos and online demo code, are discussed below.)

Deploying
=========

To Dev
------

Once all code has been compiled, you can deploy from local to dev by running:

.. code-block:: bash

   python titans/website/deploy.py

This will only deploy the HTML/JS/CSS files, and will NOT deploy other website
files (e.g. online demo files, video files).

To Production
-------------

After reviewing the files on the dev fileserver, you can deploy from dev to
production by running

.. code-block:: bash

   python titans/website/deploy.py --prod

Unlike the dev deployment code, this will copy over all files from dev to
production, including demo files and video files. (The rationale is that, when
you move from local to dev, with this script, you're testing out HTML code;
and, when you move from dev to prod, you are testing out all files and code on
the server.)

***********
Online Demo
***********

Code for the online demo lives in :code:`titans/demo`. To upload to the dev
fileserver, simply run:

.. code-block:: bash

   python titans/demo

This code was written in Phaser 3. Potential future upgrades include upgrading
the AI system; enabling P2P versus, and including additional cards. However, as
the goal of this demo is to serve as a try-before-you-buy, this does NOT have a
high priority.

******
Videos
******

Several blender videos have been created to demo this project. These use
Blender to render, and have been setup to work using Azure Batch, which allows
for massively-parallel rendering. (At one point, these were all running
sequentially on my desktop, and would take several weeks to finish; now, they
complete in mere hours.)

A Dockerfile has been provided for use by Azure Batch, so that each pod doesn't
have to install its own dependencies. To build the docker image and store it on
a container registry, run:

.. code-block:: bash

   python titans/videos/build_img.py --verbose

Once this completes, you can launch various rendering tasks via the command:

.. code-block:: bash

   python titans/videos

The main commands that should be run, in order, are:

.. code-block:: bash

   python titans/videos animate
   python titans/videos render
   python titans/videos convert

Each command should only be run after all jobs on Azure complete. Several of
these jobs string together multiple jobs that all depend on one another. (These
could have all been strung together into a single job; however, I wanted to
encourage users to look at the output of each job before moving onto the next.)

The first command, animate, creates video frames as png files. The second,
render, combines pngs to create videos. The third, convert, transforms videos
into modern codec formats.

Each of these commands has many command-line options for rerunning only certain
subsets. Refer to the documentation of each for help.

*********
Databases
*********

This repo creates and manages a MySQL database with three tables: contacts,
comments, and creds.

Comments and creds are tables meant to store user input from the website. These
were created via the python command:

.. code-block:: bash

   python titans/sql/comments.py --create
   python titans/sql/contacts.py --create

Some db management is available via flags on those arguments. Use with caution.

Various credentials are also stored in this database. This is probably not best
practice, but works, as all dev work happens on this project's cloud VM, which
is the only IP able to access the database. This credentials table was made,
and can be updated, via:

.. code-block:: bash

   python titans/sql/creds.py

Refer to the docstrings for additional information.

***
API
***

I setup a cloud VM to act as an API for this project. This API will one day be
upgraded to handle the decisions made by the AI for the online demo (which will
eventually be expanded into an online client).

The API was setup with FastAPI, and it handles all database queries and
insertions made from the website. This includes:

#. Subscribing to the email list (which adds a user's name and email to the
   contacts table)
#. Unsubscribing (removing name/email)
#. Commenting (saving comment, and optionally email, to the comments table)

Entries and scanned, sanitizied, and validating before insertions take place.

Emails are additionally automatically sent by the API whenever a comment is
left on the website, or whenever a new subscriber joins the email list.

The API can be accessed `here <https://titansapi.eastus.cloudapp.azure.com/>`_.
FastAPI docs for this API are located `here
<https://titansapi.eastus.cloudapp.azure.com/docs>`_.

Updates
=======

If you want to test changes to the API before going live, you can update the
code, and then run (on the webserver):

.. code-block:: bash

   webserver/test-api.sh

This will spin up a copy of the API on the webserver at the port 1024, which
can be accessed `here <https://titansapi.eastus.cloudapp.azure.com:1024>`_.

To update the real API, push your changes to GitHub, then run:

.. code-block:: bash

   webserver/run-service.sh

This command is automatically run every time the webserver restarts.

******
Emails
******

This package handles sending emails to all of our subscribers, all from the
command line. This uses the MS 365 Graph API to execute sending.

Emails are additionally automatically sent whenever a comment is left on the
website, or whenever a new subscriber joins the email list.

Authenticating
==============

For first-time use, you must authenticate with Office365 by following these
steps:

In the Azure portal:

#. Go to :code:`Azure Active Directory` -> :code:`App Registrations`

#. Create a new app registration. Set the redirect URI to
   :code:`http://localhost`.

#. Go to :code:`Certificates & secrets`, then create a new client secret. Copy
   the :code:`Value`, as you'll only be able to see this once.

#. Go to :code:`API Permissions`, and from Microsoft Graph the Delegated
   Permission of Mail.Send.

#. Create a :code:`~/secrets/titans-email-creds` file that looks like:

   .. code-block:: json
      
      {
          "tenant": "<directory (tenant) id>",
          "client_id": "<application (client) id>",
          "client_secret": "<certifiacte & secrets value>",
      }

In this repo:

#. Make sure you have a :code:`SECRETS_DIR` environmental variable set, e.g.
   :code:`~/secrets`.

#. Run :code:`titans/email/get-code.sh`. Follow the website it points you to,
   and authenticate with your office account. It'll redirect you to an error
   page. On that page, look at the url, and copy the :code:`code` parameter
   from the URL (i.e. the portion that reads :code:`&code=...&`). Paste that
   code into a :code:`$SECRETS_DIR/titans-email-code` file in this repo.

#. Run `auth/get-token.sh`. This will create a
   :code:`$SECRETS_DIR/titans-email-token` file, containing your
   authentication token.

#. Finally, update the :code:`creds` table in sql by running:
   
   .. code-block:: bash

      python titans/sql/creds.py

Sending Emails
==============

To send emails, run:

.. code-block:: bash

   python titans/email yamlconfig

where :code:`yamlconfig` is a configuration yaml file that is unpacked to
initialize :code:`titans.email.sender.SendEmails`. An example configuration
yaml file would look like:

.. code-block:: yaml

   subject: We're Back!
   body: 2022-07/body.html
   attachments: [
     2022-07/box.png,
     2022-07/divider.png,
     2022-07/final_judgment.png,
     2022-07/logo.png,
   ]

where :code:`2022-07/body.html` is the name of an html file you want to send,
and the attachments are names of attachments you want to attach.

Please note that any attachments can be referenced and inserted in :code:`body`
via their cid:

.. code-block:: html

   <img src="cid:logo">

and the string :code:`#|EMAIL|#` will be replaced with the receipient's email
address.

Dev: Testing
============

To send a test email, run:

.. code-block:: bash

   python tests/titans/api/subscribe_test.py

Please note that this test will fail unless it's run on the whitelisted VM!

Polls
=====

Periodically, we run polls out of the email service. To do so:

#. Create a new table in the database:

   .. code-block:: bash
   
      python titans/sql/polls.py table_name

   This table will store the poll responses.

#. Build the email that will contain this poll. This email needs to have links
   that the recipient can click on that query the API. These responses have to
   look like:

   .. code-block:: text

      https://titansapi.eastus.cloudapp.azure.com/poll/POLL_NAME?email=#|EMAIL|#&response=RESPONSE

   which includes the parameters:

   #. :code:`POLL_NAME``: The name of the poll, which is the name of the table
      created in the previous step.

   #. :code:`EMAIL`: The email of the responder, which will be auto-inserted by
      the email client. (The string :code:`#|EMAIL|#` is a special string that
      is processed by the email client.)

   #. :code:`RESPONSE`: The response to the poll, which should be hard-coded
      for each clickable link.

#. Send the email, and check the results as they roll in!
