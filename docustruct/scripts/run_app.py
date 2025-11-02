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
    # If the chosen app file doesn't exist (e.g., basic app removed),
    # fallback to the professional app.
    if not os.path.exists(app_path):
        app_file = "app_professional.py"
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
    # Ensure the top-level project root is on PYTHONPATH so imports like
    # `import docustruct` work when Streamlit changes the working directory.
    repo_root = os.path.dirname(os.path.dirname(cur_dir))
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{repo_root}:{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else repo_root
    )
    subprocess.run(cmd, env=env, cwd=repo_root)


if __name__ == "__main__":
    main()
