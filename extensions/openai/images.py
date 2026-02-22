"""
OpenAI-compatible image generation using local diffusion models.
"""

from extensions.openai.errors import ServiceUnavailableError


def generations(request):
    """
    Generate images using the loaded diffusion model.
    Returns dict with 'created' timestamp and 'data' list of images.
    """
    raise ServiceUnavailableError("Not implemented")
