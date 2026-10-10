# Example response procedure — synthetic task data

For a tier-1 API, propose rollback when error rate exceeds 5% for five minutes
and the preceding release was healthy. The service owner must approve rollback.
An agent may recommend actions but must not deploy or roll back automatically.

Before rollback: preserve the current release ID, error window and logs; verify
that the preceding release is available and database migrations are compatible.
After rollback: observe errors below 1% for ten minutes, then confirm checkout.

Payments owner: Maya. Backup contact: on-call platform engineer.
If errors persist after rollback, escalate for dependency and database diagnosis.
