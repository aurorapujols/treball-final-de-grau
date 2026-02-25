import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Data Preprocessing")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["enhance_images", "labeling", "preprocess_incoming", "show_zip_contents"],
        help="Which experiment to run"
    )

    args = parser.parse_args()

    # Dispatch to the correct experiment
    if args.task == "enhance_images":
        os.system("python -u -m pipelines.enhance_images")

    elif args.task == "labeling":
        os.system("python -u -m pipelines.labeling")

    elif args.task == "preprocess_incoming":
        os.system("python -u -m pipelines.preprocess_incoming")

    elif args.task == "show_zip_contents":
        os.system("python -u -m pipelines.zip_contents")

if __name__ == "__main__":

    main()