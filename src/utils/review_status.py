from typing import Optional


def normalize_status(status: Optional[str]) -> str:
    if not status:
        return "APPROVE_WITH_WARNINGS"
    upper = str(status).strip().upper()
    if upper == "RESOLVED":
        return "APPROVE_WITH_WARNINGS"
    if upper == "APPROVED":
        return "APPROVED"
    if upper == "REJECTED":
        return "REJECTED"
    return "APPROVE_WITH_WARNINGS"
