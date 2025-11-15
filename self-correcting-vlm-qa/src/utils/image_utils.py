"""
Image utility functions for processing and compression.
"""
import base64
from io import BytesIO
from PIL import Image
from loguru import logger


def resize_image_if_needed(image_base64: str, max_size_mb: float = 4.8, preserve_quality: bool = True) -> str:
    """
    Resize image if it exceeds the maximum size, preserving quality as much as possible.

    Args:
        image_base64: Base64-encoded image (with or without data URL prefix)
        max_size_mb: Maximum size in megabytes (default 4.8MB, safe under 5MB limit)
        preserve_quality: If True, try harder to preserve quality with minimal resizing

    Returns:
        Resized base64-encoded image
    """
    # Remove data URL prefix if present
    if "," in image_base64:
        prefix, image_data = image_base64.split(",", 1)
    else:
        prefix = None
        image_data = image_base64

    # Decode image
    image_bytes = base64.b64decode(image_data)
    decoded_size_mb = len(image_bytes) / (1024 * 1024)

    # Check the actual base64 size that will be sent to the VLM
    base64_size_mb = len(image_data) / (1024 * 1024)

    logger.info(f"Image size - Decoded: {decoded_size_mb:.2f}MB, Base64: {base64_size_mb:.2f}MB")
    logger.info(f"Max allowed size: {max_size_mb}MB")

    # Check base64 size (what the VLM actually receives)
    if base64_size_mb <= max_size_mb:
        logger.info("Image base64 is within size limit, returning as-is")
        # Ensure it has proper data URL prefix with correct format
        if not image_base64.startswith("data:image"):
            # Detect format from image bytes
            img = Image.open(BytesIO(image_bytes))
            if img.format:
                format_lower = img.format.lower()
                logger.info(f"Detected image format: {format_lower}")
                return f"data:image/{format_lower};base64,{image_data}"
            else:
                logger.info("Could not detect format, defaulting to jpeg")
                return f"data:image/jpeg;base64,{image_data}"
        # Already has data URL prefix, return as-is
        logger.info(f"Image already has data URL prefix: {image_base64[:50]}...")
        return image_base64

    logger.warning(f"Image exceeds limit (base64: {base64_size_mb:.2f}MB > {max_size_mb}MB), processing...")
    current_size_mb = decoded_size_mb  # Use decoded size for resize calculations

    # Load image
    img = Image.open(BytesIO(image_bytes))

    # Convert RGBA to RGB if needed (before resizing to preserve quality)
    if img.mode == 'RGBA':
        # Create white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
        img = background

    buffered = BytesIO()

    if preserve_quality:
        # Try compression first without resizing
        quality = 95
        while quality >= 75:
            buffered.seek(0)
            buffered.truncate()
            img.save(buffered, format="JPEG", quality=quality, optimize=True)

            if len(buffered.getvalue()) / (1024 * 1024) <= max_size_mb:
                # Success with compression only!
                final_size_mb = len(buffered.getvalue()) / (1024 * 1024)
                logger.info(f"Compressed without resize: {decoded_size_mb:.2f}MB → {final_size_mb:.2f}MB (quality={quality})")
                resized_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                # Return with JPEG prefix since we converted to JPEG
                return f"data:image/jpeg;base64,{resized_base64}"
            quality -= 5

    # Need to resize - calculate minimal resize ratio
    ratio = (max_size_mb / current_size_mb) ** 0.5
    # Use slightly more aggressive resize to have room for quality
    ratio *= 0.95

    new_width = int(img.width * ratio)
    new_height = int(img.height * ratio)

    # Keep reasonable minimum size
    min_dimension = 800
    if new_width < min_dimension or new_height < min_dimension:
        scale = min_dimension / min(new_width, new_height)
        new_width = int(new_width * scale)
        new_height = int(new_height * scale)

    # Resize with high-quality algorithm
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Save with high quality
    buffered.seek(0)
    buffered.truncate()
    quality = 92  # Start with high quality

    while quality > 60:
        buffered.seek(0)
        buffered.truncate()
        img_resized.save(buffered, format="JPEG", quality=quality, optimize=True)

        if len(buffered.getvalue()) / (1024 * 1024) <= max_size_mb:
            break
        quality -= 3

    # Encode to base64
    resized_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    final_size_mb = len(buffered.getvalue()) / (1024 * 1024)
    logger.info(f"Image processed successfully: {decoded_size_mb:.2f}MB → {final_size_mb:.2f}MB")

    # Return with JPEG prefix since we converted to JPEG
    # This ensures the VLM receives the correct media type
    return f"data:image/jpeg;base64,{resized_base64}"
