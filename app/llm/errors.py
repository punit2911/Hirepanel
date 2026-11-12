class LLMUnavailable(Exception):
    """No provider could produce a valid answer. Agents catch this and fall back to heuristics."""


class ProviderError(Exception):
    pass


class RateLimited(ProviderError):
    def __init__(self, msg: str = "rate limited", retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


class TransientError(ProviderError):
    """Timeouts, 5xx, empty replies: worth retrying."""


class PermanentError(ProviderError):
    """Bad key, unknown model, malformed request: do not retry this provider for a while."""


class DailyQuotaExhausted(ProviderError):
    pass
