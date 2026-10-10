# Example service ownership — synthetic task data

| Service | Owner | Tier | Deployment |
|---|---|---|---|
| payments-api | Maya | 1 | payments-2026.10.10-3 |
| search-api | Arjun | 2 | search-2026.10.09-7 |
| notification-worker | Leena | 2 | notify-2026.10.10-1 |

Payments handles checkout authorizations. Search affects catalog discovery.
Notification sends receipts after a successful payment and can replay its queue.
