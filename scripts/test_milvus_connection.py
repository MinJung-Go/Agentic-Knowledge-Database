#!/usr/bin/env python3
"""测试 Milvus 连接 - 对比两种连接方式"""
import sys
sys.path.insert(0, ".")

from configs.settings import settings


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def test_connections_api():
    """方式一：使用 pymilvus.connections API (gRPC)"""
    print_section("方式一: connections.connect (gRPC)")

    try:
        from pymilvus import connections, utility

        # 从配置中获取 host（去掉 http:// 前缀）
        host = settings.milvus_host.replace("http://", "").replace("https://", "")
        port = settings.milvus_port

        print(f"  Host: {host}")
        print(f"  Port: {port}")
        print(f"  连接中...")

        connections.connect(
            alias="test_grpc",
            host=host,
            port=port,
            timeout=10,
        )

        print("  ✅ 连接成功!")

        # 获取版本信息
        try:
            version = utility.get_server_version(using="test_grpc")
            print(f"  Milvus 版本: {version}")
        except Exception as e:
            print(f"  ⚠️ 获取版本失败: {e}")

        # 列出 collections
        collections = utility.list_collections(using="test_grpc")
        print(f"  Collections 数量: {len(collections)}")
        if collections:
            print(f"  Collections: {collections[:5]}")

        connections.disconnect("test_grpc")
        return True

    except Exception as e:
        print(f"  ❌ 连接失败: {e}")
        return False


def test_milvusclient_api():
    """方式二：使用 MilvusClient API (URI 方式)"""
    print_section("方式二: MilvusClient (URI)")

    try:
        from pymilvus import MilvusClient

        # 构建 URI
        host = settings.milvus_host.replace("http://", "").replace("https://", "")
        port = settings.milvus_port
        uri = f"http://{host}:{port}"

        print(f"  URI: {uri}")
        print(f"  连接中...")

        client = MilvusClient(uri=uri)

        print("  ✅ 连接成功!")

        # 列出 collections
        collections = client.list_collections()
        print(f"  Collections 数量: {len(collections)}")
        if collections:
            print(f"  Collections: {collections[:5]}")

            # 测试获取 collection 统计信息
            for coll_name in collections[:1]:
                try:
                    stats = client.get_collection_stats(collection_name=coll_name)
                    print(f"  Collection '{coll_name}' 行数: {stats.get('row_count', 'N/A')}")
                except Exception as e:
                    print(f"  ⚠️ 获取统计失败: {e}")

        return True

    except Exception as e:
        print(f"  ❌ 连接失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_milvusclient_with_token():
    """方式三：使用 MilvusClient API (带 token)"""
    print_section("方式三: MilvusClient (带空 token)")

    try:
        from pymilvus import MilvusClient

        host = settings.milvus_host.replace("http://", "").replace("https://", "")
        port = settings.milvus_port
        uri = f"http://{host}:{port}"

        print(f"  URI: {uri}")
        print(f"  Token: '' (空)")
        print(f"  连接中...")

        # 使用空 token（与用户代码一致）
        client = MilvusClient(uri=uri, token="")

        print("  ✅ 连接成功!")

        # 测试基本操作
        has_coll = client.has_collection("test_collection_12345")
        print(f"  has_collection 测试: {has_coll}")

        collections = client.list_collections()
        print(f"  Collections 数量: {len(collections)}")

        return True

    except Exception as e:
        print(f"  ❌ 连接失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print_section("Milvus 连接测试")
    print(f"\n配置信息 (来自 configs/settings.py):")
    print(f"  milvus_host: {settings.milvus_host}")
    print(f"  milvus_port: {settings.milvus_port}")

    results = {}

    # 测试三种连接方式
    results["connections API"] = test_connections_api()
    results["MilvusClient API"] = test_milvusclient_api()
    results["MilvusClient + token"] = test_milvusclient_with_token()

    # 汇总结果
    print_section("测试结果汇总")
    for name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {name}: {status}")

    if all(results.values()):
        print("\n✅ 所有连接方式都成功！Milvus 服务正常。")
        return 0
    elif any(results.values()):
        print("\n⚠️ 部分连接方式失败，请检查上方错误信息。")
        return 1
    else:
        print("\n❌ 所有连接方式都失败，请检查 Milvus 服务状态和网络配置。")
        return 2


if __name__ == "__main__":
    sys.exit(main())
