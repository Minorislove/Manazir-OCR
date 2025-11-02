import os
import subprocess
import sys


def main():
    argv = sys.argv[1:]
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    # Determine which app file to launch
    # Priority: --app arg > DOCUSTRUCT_APP_FILE > DOCUSTRUCT_APP
    app_file = None
    next_is_value = False
    forwarded_args = []
    for arg in argv:
        if next_is_value:
            app_file = arg
            next_is_value = False
            continue
        if arg == "--app":
            next_is_value = True
            continue
        forwarded_args.append(arg)

    if app_file is None:
        app_file = os.environ.get("DOCUSTRUCT_APP_FILE")

    if app_file is None:
        app_mode = os.environ.get("DOCUSTRUCT_APP", "basic").lower()
        if app_mode in ("pro", "professional"):
            app_file = "app_professional.py"
        else:
            # default and any legacy 'enhanced' value fall back to basic app
            app_file = "app.py"

    app_path = os.path.join(cur_dir, app_file)

    cmd = [
        "streamlit",
        "run",
        app_path,
        "--server.fileWatcherType",
        "none",
        "--server.headless",
        "true",
    ]
    if forwarded_args:
        cmd += ["--"] + forwarded_args
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
