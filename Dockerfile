FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY trimtab ./trimtab
COPY engine ./engine
RUN pip install --no-cache-dir . "psycopg[binary]"
ENTRYPOINT ["python3", "-m"]
CMD ["trimtab.server", "--help"]
