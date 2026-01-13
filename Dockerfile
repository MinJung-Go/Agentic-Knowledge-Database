FROM python:3.11-slim

# 设置时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY configs/ configs/
COPY core/ core/
COPY app/ app/

# 环境变量
ENV PYTHONUNBUFFERED=1
ENV WORKERS=4

EXPOSE 8000

# 健康检查（容器运行后才执行，--start-period 给应用启动时间）
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 使用 gunicorn + uvicorn workers（生产环境推荐）
CMD ["sh", "-c", "gunicorn app:app -w ${WORKERS} -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000"]
