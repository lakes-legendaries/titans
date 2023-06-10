from os.path import dirname, join, realpath

import typer

import titans.cloud


# create cli
app = typer.Typer()


# option to upload
@app.command()
def upload():
    """Upload demo files to dev server"""
    folder = dirname(realpath(__file__))
    titans.cloud.upload(join(folder, "demo"), "--pattern", "*.js", dest="demo")
    titans.cloud.upload(join(folder, "titans-demo.html"))


# run cli
if __name__ == "__main__":
    app()
