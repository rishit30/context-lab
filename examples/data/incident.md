# Example observations — synthetic task data

At 14:05 payments-api error rate reached 8.2% for six consecutive minutes.
Its previous deployment had 0.4% errors. The incident started immediately after
payments-2026.10.10-3. The last known good release is payments-2026.10.09-2.

search-api has p95 latency of 480 ms, below its 600 ms alert threshold.
notification-worker has a queue age of 45 seconds, below its 120 second threshold.

These observations identify correlation, not a confirmed root cause.
No customer count or financial impact estimate has been supplied.
