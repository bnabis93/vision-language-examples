"""Diffusers inference example
- Author: Bono (qhsh9713@gmail.com)
"""
from diffusers import DiffusionPipeline


def parse():
    """
    Parse arguments for the inference.
    """
    parser = argparse.ArgumentParser(
        description="Stable Diffusion Model Inference Example"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run the model on."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    # Pipelien example
    pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipeline.to(args.device)

    output = pipeline("An image of a squirrel in Picasso style").images[0]
    output.save("image_of_squirrel_painting.png")
