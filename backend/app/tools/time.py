from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .base import Tool


class TimeTool(Tool):
    name = "get_time"
    description = "Returns the current date and time for a given IANA timezone (defaults to UTC)."
    parameters = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "IANA timezone name, e.g. 'Asia/Ho_Chi_Minh' or 'America/New_York'.",
            }
        },
        "required": [],
    }

    def run(self, arguments: dict[str, Any]) -> dict[str, str]:
        timezone = arguments.get("timezone", "UTC")
        try:
            tz = ZoneInfo(timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown timezone: {timezone!r}") from exc

        now = datetime.now(tz)
        return {
            "timezone": timezone,
            "iso": now.isoformat(),
            "formatted": now.strftime("%A, %B %d, %Y %I:%M %p"),
        }
