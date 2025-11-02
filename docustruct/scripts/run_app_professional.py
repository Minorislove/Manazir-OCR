"""
Thin wrapper to launch the professional Streamlit app via the unified runner.
"""

import os


def main():
    os.environ["DOCUSTRUCT_APP"] = "professional"
    from docustruct.scripts.run_app import main as run

    run()


if __name__ == "__main__":
    main()
