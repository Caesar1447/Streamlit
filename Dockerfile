FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN adduser --disabled-password --gecos "" appuser
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=appuser:appuser . .
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python -c 'import urllib.request; urllib.request.urlopen("http://localhost:8501/_stcore/health",timeout=3)' || exit 1
CMD ["streamlit", "run", "Dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
