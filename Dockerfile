FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# glob, 不是逐个列举：新增模块时不会漏掉
COPY rh_*.py ./

# 固定 uid/gid，好让宿主机上 bind 挂载的 credentials.json chown 到同一个 id。
RUN useradd --uid 10001 --user-group --no-create-home --shell /usr/sbin/nologin app \
 && chown -R app:app /app
USER app

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8002/health', timeout=4).status == 200 else 1)"]

# rh_server.main() 绑的是 127.0.0.1，在容器里外部访问不到，所以直接跑 uvicorn 绑 0.0.0.0。
CMD ["uvicorn", "rh_server:app", "--host", "0.0.0.0", "--port", "8002"]
