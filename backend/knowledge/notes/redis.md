# Redis Notes

Redis can implement rate limiting using token bucket algorithms. Each client gets a
bucket of tokens that refills at a fixed rate; a request is allowed only if a token is
available, and denied (or queued) otherwise.

This works well for API rate limiting because the check-and-decrement operation can be
done atomically in Redis (e.g. via a Lua script or `INCR` + `EXPIRE`), so it holds up
under concurrent requests across multiple app servers.
