#!/usr/bin/env python3
"""检查 settings.py 中配置的服务是否可用"""
import socket
import sys
from urllib.parse import urlparse

import httpx

# 添加项目根目录到 path
sys.path.insert(0, ".")

from configs.settings import settings


def check_tcp(host: str, port: int, timeout: float = 3.0) -> tuple[bool, str]:
    """检查 TCP 端口是否可达"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, "连接成功"
    except socket.timeout:
        return False, "连接超时"
    except ConnectionRefusedError:
        return False, "连接被拒绝"
    except OSError as e:
        return False, str(e)


def check_http(url: str, timeout: float = 5.0) -> tuple[bool, str]:
    """检查 HTTP 服务是否可用"""
    try:
        # 尝试常见的健康检查端点
        for path in ["", "/health", "/v1/models", "/api/health"]:
            try:
                test_url = url.rstrip("/") + path
                response = httpx.get(test_url, timeout=timeout)
                if response.status_code < 500:
                    return True, f"HTTP {response.status_code} ({path or '/'})"
            except Exception:
                continue
        return False, "所有端点均不可达"
    except httpx.TimeoutException:
        return False, "请求超时"
    except httpx.ConnectError as e:
        return False, f"连接失败: {e}"
    except Exception as e:
        return False, str(e)


def parse_url(url: str) -> tuple[str, int]:
    """解析 URL 获取 host 和 port"""
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return host, port


def main():
    print("=" * 60)
    print("服务可用性检查")
    print("=" * 60)

    services = [
        {
            "name": "MinerU (文档解析)",
            "url": settings.mineru_base_url,
            "type": "http",
        },
        {
            "name": "LLM (vLLM)",
            "url": settings.llm_base_url,
            "type": "http",
        },
        {
            "name": "Embedding",
            "url": settings.embedding_base_url,
            "type": "http",
        },
        {
            "name": "Rerank",
            "url": settings.rerank_base_url,
            "type": "http",
        },
        {
            "name": "Milvus",
            "host": settings.milvus_host.replace("http://", "").replace("https://", ""),
            "port": settings.milvus_port,
            "type": "tcp",
        },
    ]

    results = []

    for svc in services:
        name = svc["name"]

        if svc["type"] == "http":
            url = svc["url"]
            host, port = parse_url(url)

            # 先检查 TCP
            tcp_ok, tcp_msg = check_tcp(host, port)

            if tcp_ok:
                # TCP 通了再检查 HTTP
                http_ok, http_msg = check_http(url)
                status = "✅" if http_ok else "⚠️"
                msg = http_msg
                ok = http_ok
            else:
                status = "❌"
                msg = tcp_msg
                ok = False

            print(f"\n{status} {name}")
            print(f"   地址: {url}")
            print(f"   状态: {msg}")

        else:  # tcp
            host = svc["host"]
            port = svc["port"]
            ok, msg = check_tcp(host, port)
            status = "✅" if ok else "❌"

            print(f"\n{status} {name}")
            print(f"   地址: {host}:{port}")
            print(f"   状态: {msg}")

        results.append((name, ok))

    # 汇总
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)

    ok_count = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        print(f"  {'✅' if ok else '❌'} {name}")

    print(f"\n可用: {ok_count}/{total}")

    if ok_count < total:
        print("\n⚠️  部分服务不可用，请检查配置或启动相关服务")
        return 1
    else:
        print("\n✅ 所有服务均可用")
        return 0


if __name__ == "__main__":
    sys.exit(main())
