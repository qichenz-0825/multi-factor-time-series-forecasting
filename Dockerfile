FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir --no-deps .
ENTRYPOINT ["forecasting"]
CMD ["demo", "--output", "/app/reports/demo"]
