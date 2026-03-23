import os
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Data Preprocessing")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["enhance_images", "labeling", "preprocess_incoming", "show_zip_contents", "temp"],
        help="Which experiment to run"
    )

    args = parser.parse_args()

    print("Received task: ", args.task)

    # Dispatch to the correct experiment
    if args.task == "enhance_images":
        subprocess.run(["python", "-u", "-m", "pipelines.enhance_images"], check=True)

    elif args.task == "labeling":
        subprocess.run(["python", "-u", "-m", "pipelines.labeling"], check=True)

    elif args.task == "preprocess_incoming":
        subprocess.run(["python", "-u", "-m", "pipelines.preprocess_incoming"], check=True)

    elif args.task == "show_zip_contents":
        subprocess.run(["python", "-u", "-m", "pipelines.zip_contents"], check=True)

    elif args.task == "temp":
        subprocess.run(["python", "-u", "-m", "pipelines.temp"], check=True)

if __name__ == "__main__":

    main()