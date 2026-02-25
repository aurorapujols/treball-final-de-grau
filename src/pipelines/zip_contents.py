import os
import py7zr

from config import config

def get_monthly_contents(videos_root, metadata_root):
    # List all .7z archives
    video_archives = sorted([f for f in os.listdir(videos_root) if f.endswith(".7z")])
    meta_archives = sorted([f for f in os.listdir(metadata_root) if f.endswith(".7z")])

    print("\n=== VIDEO ARCHIVES ===")
    process_archives(video_archives, videos_root, filetype="AVI")

    print("\n=== METADATA ARCHIVES ===")
    process_archives(meta_archives, metadata_root, filetype="XML")


def process_archives(archives, root, filetype):
    for archive in archives:
        month_name = os.path.splitext(archive)[0]  # e.g. "October", "202510"
        archive_path = os.path.join(root, archive)

        print(f"\nFolder '{archive}' (Month: {month_name}):")

        with py7zr.SevenZipFile(archive_path, mode="r") as z:
            # Ignore directory entries (7z marks them with a trailing slash)
            names = [ n for n in z.getnames() if not n.endswith("/") and "." in os.path.basename(n) ]

            # Count only files of the expected type
            count = sum(1 for n in names if n.lower().endswith(filetype.lower()))

            print(f"    {filetype} files: {count}")

            # Detect unexpected files
            unexpected = [
                n for n in names
                if not n.lower().endswith(filetype.lower())
            ]

            if unexpected:
                print("    ⚠ Unexpected files found:")
                for u in unexpected:
                    print(f"        {u}")


if __name__ == "__main__":
    print("Content of the zip folders in /videos and /metadata:")

    videos_root = config.paths.raw_videos_root
    metadata_root = config.paths.raw_metadata_root

    get_monthly_contents(videos_root, metadata_root)

    print("\nDONE!")
