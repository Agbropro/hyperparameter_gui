"""User-submitted issue and suggestion tickets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from uuid import uuid4

from domain.entities import utc_now


class TicketType(str, Enum):
    FEATURE = "feature"
    BUG = "bug"
    MISC = "misc"


@dataclass
class Ticket:
    title: str
    type: TicketType
    message: str
    page: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    status: str = "new"
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)
