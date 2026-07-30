FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn flask flask-caching lightgbm scikit-learn plotly
COPY . .
ENV PYTHONPATH=/app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "website.app:create_app()"]
