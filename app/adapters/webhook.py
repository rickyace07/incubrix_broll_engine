import asyncio
import logging
import httpx
from typing import Dict, Any, Optional

logger = logging.getLogger("broll.webhook")

async def dispatch_webhook(
    webhook_url: Optional[str],
    payload: Dict[str, Any],
    max_retries: int = 3,
    base_delay: float = 1.5
) -> bool:
    """Dispatches job completion payload to client webhook using exponential backoff."""
    if not webhook_url:
        return False

    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(1, max_retries + 1):
            try:
                response = await client.post(webhook_url, json=payload)
                if response.status_code in (200, 201, 202, 204):
                    logger.info(f"Webhook delivered to {webhook_url} on attempt {attempt}")
                    return True
                logger.warning(
                    f"Webhook attempt {attempt} returned status {response.status_code}"
                )
            except Exception as exc:
                logger.warning(f"Webhook attempt {attempt} failed: {exc}")

            if attempt < max_retries:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))

    logger.error(f"Webhook delivery failed after {max_retries} attempts to {webhook_url}")
    return False