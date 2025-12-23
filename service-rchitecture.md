# API 系统架构

## 整体架构

奇境探索 API 采用现代化的微服务架构设计，基于 FastAPI 框架构建，提供高性能、可扩展的异步服务。

```mermaid
graph TB
    subgraph "客户端层"
        A[iOS App]
        B[Android App]
        C[Web App]
        D[Desktop App]
    end

    subgraph "网关层"
        E[Nginx/Cloudflare]
        F[负载均衡器（暂无，未来扩展目标）]
    end

    subgraph "应用层"
        G[Questech API]
        H[静态文件服务]
    end

    subgraph "缓存层"
        I[Redis]
    end

    subgraph "数据层"
        J[MySQL 主库]
        K[MySQL 从库]
        L[文件存储]
    end

    subgraph "外部服务"
        M[OpenAI API]
        N[Apple ID]
        O[短信服务]
        P[AI 模型服务]
    end

    A --> E
    B --> E
    C --> E
    D --> E
    E --> F
    F --> G
    F --> H
    G --> I
    G --> J
    G --> K
    G --> L
    G --> M
    G --> N
    G --> O
    G --> P
```

## 技术栈

### 核心框架
- **FastAPI 0.116.1** - 现代、快速的 Web 框架
- **Python 3.9+** - 主要编程语言
- **Uvicorn** - ASGI 服务器
- **Pydantic 2.x** - 数据验证和序列化

### 数据存储
- **MySQL 8.0+** - 主数据库
- **Redis 5.0+** - 缓存和会话存储
- **文件系统** - 静态文件和用户上传文件

### 异步支持
- **asyncio** - Python 异步编程
- **SQLAlchemy 2.0** - 异步 ORM
- **aiomysql** - 异步 MySQL 驱动
- **aioredis** - 异步 Redis 客户端

### 认证和安全
- **JWT (JSON Web Token)** - 令牌认证
- **Apple Sign-In** - iOS 认证集成
- **bcrypt** - 密码哈希
- **CORS** - 跨域资源共享

### 外部集成
- **OpenAI API** - GPT 模型服务
- **DeepSeek API** - 本地模型服务
- **阿里云短信** - 短信验证码服务
- **Apple ID** - 登录认证

## 架构原则

### 1. 模块化设计
```python
# 应用模块结构
apps/
├── admin/          # 管理后台模块
├── chat/           # 聊天服务模块
├── common/         # 通用服务模块
├── config/         # 配置管理模块
├── content/        # 内容管理模块
├── intelligent/    # AI 智能体模块
├── toolkit/        # 工具包模块
└── user/           # 用户管理模块
```

每个模块都是独立的 FastAPI 路由，职责单一，便于维护和扩展。

### 2. 异步优先
所有 I/O 操作都采用异步模式，确保高并发性能：
```python
@router.post("/chat/completions")
async def create_chat_completion(request: ChatRequest):
    # 异步数据库查询
    user = await get_user_info(request.user_id)

    # 异步 API 调用
    response = await call_openai_api(request.messages)

    # 异步保存聊天记录
    await save_chat_message(request.user_id, response)

    return response
```

### 3. 配置驱动
使用 Pydantic Settings 管理配置，支持多环境：
```python
# shared/config.py
class Settings(BaseSettings):
    # 数据库配置
    mysql_url: str = "mysql+pymysql://user:pass@localhost:3306/questech"

    # Redis 配置
    redis_url: str = "redis://localhost:6379/1"

    # JWT 配置
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"

    # 环境检测
    service_debug: bool = False

    class Config:
        env_file = ".env"
```

### 4. 中间件管道
全局中间件处理通用逻辑：
```python
# main.py
app.add_middleware(CORSMiddleware, ...)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuthMiddleware)
```

## 核心组件

### 1. 路由管理器 (main.py)
负责应用初始化和路由注册：
```python
from fastapi import FastAPI
from apps.chat import router as chat_router
from apps.user import router as user_router

app = FastAPI(title="Questech API", version="1.0.0")

# 注册路由
app.include_router(chat_router, prefix="/chat", tags=["chat"])
app.include_router(user_router, prefix="/user", tags=["user"])
```

### 2. 中间件系统 (utils/middleware.py)

#### 请求日志中间件
```python
class RequestLoggingMiddleware:
    async def __call__(self, request: Request, call_next):
        start_time = time.time()

        response = await call_next(request)

        process_time = time.time() - start_time
        logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.4f}s")

        return response
```

#### 认证中间件
```python
class AuthMiddleware:
    async def __call__(self, request: Request, call_next):
        # 检查是否需要认证
        if self.is_public_path(request.url.path):
            return await call_next(request)

        # 验证 JWT 令牌
        token = self.extract_token(request)
        if not token or not await self.validate_token(token):
            raise HTTPException(status_code=401, detail="Invalid token")

        return await call_next(request)
```

### 3. 会话管理器 (utils/session_manager.py)
处理用户会话和令牌：
```python
class SessionManager:
    def __init__(self, redis_client):
        self.redis = redis_client

    async def create_session(self, user_id: int, access_token: str):
        # 存储会话信息
        session_key = f"session:{user_id}"
        await self.redis.hset(session_key, mapping={
            "access_token": access_token,
            "created_at": datetime.utcnow().isoformat(),
            "last_active": datetime.utcnow().isoformat()
        })
        await self.redis.expire(session_key, 86400)  # 24小时过期

    async def validate_session(self, user_id: int, access_token: str):
        session_key = f"session:{user_id}"
        stored_token = await self.redis.hget(session_key, "access_token")
        return stored_token == access_token
```

### 4. 数据库连接池
使用 SQLAlchemy 异步连接池：
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

engine = create_async_engine(
    settings.mysql_url,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
```

## 数据流架构

### 请求处理流程
```mermaid
sequenceDiagram
    participant Client as 客户端
    participant LB as 负载均衡
    participant API as FastAPI 应用
    participant MW as 中间件
    participant Auth as 认证服务
    participant DB as 数据库
    participant Cache as Redis 缓存

    Client->>LB: HTTP 请求
    LB->>API: 转发请求
    API->>MW: 中间件链处理
    MW->>Auth: JWT 验证
    Auth->>Cache: 会话检查
    Cache-->>Auth: 会话信息
    Auth-->>MW: 验证结果
    MW->>DB: 业务逻辑处理
    DB-->>MW: 数据结果
    MW-->>API: 处理完成
    API-->>Client: HTTP 响应
```

### 聊天请求处理
```mermaid
sequenceDiagram
    participant Client as 客户端
    participant API as FastAPI
    participant DB as 数据库
    participant LLM as AI 模型
    participant Cache as Redis

    Client->>API: 聊天请求 (带 JWT)
    API->>DB: 验证用户积分
    DB-->>API: 积分信息
    API->>LLM: 调用 AI 模型
    LLM-->>API: 模型响应
    API->>DB: 保存聊天记录
    API->>DB: 扣除积分
    API->>Cache: 更新用户状态
    API-->>Client: 返回聊天响应
```

## 性能优化

### 1. 数据库优化
- **连接池**: 复用数据库连接
- **索引优化**: 关键查询字段建立索引
- **读写分离**: 查询使用从库
- **分页查询**: 避免大结果集

### 2. 缓存策略
```python
# 模型列表缓存
async def get_available_models():
    cache_key = "models:list"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    models = await fetch_models_from_config()
    await redis.setex(cache_key, 3600, json.dumps(models))
    return models
```

### 3. 异步处理
- **并发请求**: 使用 asyncio.gather 处理多个请求
- **流式响应**: 大数据量传输使用流式
- **后台任务**: 耗时操作异步执行

### 4. 负载均衡

暂时还用不上，没有实现。

```nginx
upstream questech_api {
    server 127.0.0.1:10089 weight=3;
    server 127.0.0.1:10090 weight=2;
    server 127.0.0.1:10091 weight=1;
}

server {
    listen 80;
    server_name api.questech.com;

    location / {
        proxy_pass http://questech_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 安全架构

### 1. 认证授权
- **JWT 令牌**: 无状态认证
- **刷新令牌**: 安全的令牌更新
- **权限控制**: 基于角色的访问控制

### 2. 数据保护
```python
# 敏感数据加密
def encrypt_sensitive_data(data: str) -> str:
    cipher = AES.new(settings.encryption_key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(data.encode())
    return base64.b64encode(cipher.nonce + tag + ciphertext).decode()

# 输入验证
class UserInput(BaseModel):
    username: str = Field(..., min_length=3, max_length=20, regex="^[a-zA-Z0-9_]+$")
    email: EmailStr
    phone: str = Field(..., regex="^1[3-9]\d{9}$")
```

### 3. API 安全
- **CORS 配置**: 限制跨域访问
- **请求限流**: 防止 API 滥用
- **输入验证**: 防止注入攻击
- **HTTPS 强制**: 加密数据传输

## 监控和日志

### 1. 日志系统
```python
import structlog

logger = structlog.get_logger()

# 结构化日志
logger.info("User login",
           user_id=user.id,
           ip_address=request.client.host,
           user_agent=request.headers.get("user-agent"))

# 错误日志
logger.error("Database connection failed",
            error=str(e),
            query=query,
            params=params)
```

### 2. 性能监控
```python
# 请求性能追踪
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            logger.info(f"{func.__name__} completed", duration=duration)
    return wrapper
```

### 3. 健康检查
```python
@router.get("/health")
async def health_check():
    checks = {
        "database": await check_database(),
        "redis": await check_redis(),
        "external_apis": await check_external_apis()
    }

    if all(checks.values()):
        return {"status": "healthy", "checks": checks}
    else:
        return {"status": "unhealthy", "checks": checks}
```

## 扩展性设计

### 1. 水平扩展
- **无状态设计**: 应用层不保存状态
- **负载均衡**: 多实例部署
- **数据库分片**: 大数据量分片存储
- **微服务拆分**: 按业务域拆分服务

### 2. 插件系统
```python
class PluginManager:
    def __init__(self):
        self.plugins = {}

    def register_plugin(self, name: str, plugin: BasePlugin):
        self.plugins[name] = plugin

    async def execute_hook(self, hook_name: str, *args, **kwargs):
        for plugin in self.plugins.values():
            if hasattr(plugin, hook_name):
                await getattr(plugin, hook_name)(*args, **kwargs)

# 使用示例
plugin_manager = PluginManager()
plugin_manager.register_plugin("chat_analytics", ChatAnalyticsPlugin())
```

## 部署架构

### 开发环境
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "10089:8000"
    environment:
      - SERVICE_DEBUG=true
    volumes:
      - ./logs:/app/logs
    depends_on:
      - mysql
      - redis

  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: root
      MYSQL_DATABASE: questech

  redis:
    image: redis:6.0
```

### 生产环境
```yaml
version: '3.8'
services:
  api:
    image: questech/api:latest
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M
    environment:
      - SERVICE_DEBUG=false
      - MYSQL_URL=${MYSQL_URL}
      - REDIS_URL=${REDIS_URL}

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
```

---

*此架构文档详细说明了 奇境探索 的技术架构设计，为开发者提供了系统性的技术理解。*

**最后更新**: 2025-12-22