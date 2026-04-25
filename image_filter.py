import os

def get_latest_images(images_dir):
    if not os.path.isdir(images_dir):
        return []

    return [
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        and os.path.isfile(os.path.join(images_dir, f))
    ]