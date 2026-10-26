# Security and privacy boundaries

1. **No PHI:** Accept only generated synthetic or properly authorized public non-identifying demonstration data. Public Streamlit instances are not suitable for patient health records.
2. **Uploads:** Limit size and rows. The app parses uploads in-process for demonstration purposes. Do not log raw rows or share uploaded datasets.
3. **Inference API:** Local-only, no authentication or rate limits. Not fit for public hosting. Docker Compose binds API/dashboard to 127.0.0.1.
4. **Model artifacts:** joblib uses pickle, which can execute malicious code during loading. Only load artifacts produced by your trusted build pipeline; never accept uploaded .joblib files.
5. **Secrets:** `.env` and `.streamlit/secrets.toml` are gitignored. Put runtime secrets in the hosting provider's secret manager, not in source control.
6. **Dependencies:** Upgrade and audit dependencies before actual deployment, enable GitHub security scanning, and scan container images.
7. **Next security phase:** Identity/access management, reverse proxy, HTTPS, request rate limits, safe telemetry, penetration testing and threat modeling.
