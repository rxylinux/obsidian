---
title: VPS搭建科学上网指南
date: 2026-03-02
tags:
  - VPS
  - 代理
  - 网络
  - 技术学习
status: active
---

# VPS搭建科学上网指南

## 📋 目录

- [[#主流方案对比]]
- [[#VLESS + REALITY（推荐）]]
- [[#Hysteria2]]
- [[#客户端工具]]
- [[#常见问题]]

---

## 主流方案对比

| 方案 | 状态 | 优点 | 缺点 | 推荐度 |
|------|------|------|------|--------|
| **VLESS + REALITY** | ✅ 推荐 | 无需域名、抗封锁、稳定 | 配置稍复杂 | ⭐⭐⭐⭐⭐ |
| Hysteria2 | ⚠️ 可选 | 速度快、低延迟 | UDP易被QoS限制、敏感期弱 | ⭐⭐⭐ |
| VLESS + TLS | ⚠️ 可选 | 稳定性好 | 需要域名和证书 | ⭐⭐⭐⭐ |
| Trojan | ⚠️ 可选 | 成熟稳定 | 需要证书 | ⭐⭐⭐ |
| VMess + WS | ❌ 不推荐 | 兼容性好 | 易被封 | ⭐⭐ |

### 核心技术栈

```
┌─────────────────────────────────────────────────────────────┐
│                      协议层级结构                             │
├─────────────────────────────────────────────────────────────┤
│  应用层: VLESS | Trojan | Shadowsocks | VMess                │
│  传输层: TCP | UDP | WebSocket | HTTP/2 | gRPC               │
│  安全层: TLS | REALITY | XTLS                               │
│  混淆层: REALITY | V2Ray-Plugin                             │
└─────────────────────────────────────────────────────────────┘
```

---

## VLESS + REALITY（推荐）

### 为什么选择 VLESS + REALITY？

| 特性 | 说明 |
|------|------|
| **无域名需求** | 不需要购买域名和申请证书 |
| **抗封锁能力强** | 伪造TLS握手，流量伪装成正常HTTPS |
| **动态端口** | 支持端口跳变，避免固定端口被标记 |
| **高隐蔽性** | 使用真实网站SNI伪装（如Cloudflare、Apple） |

### 服务端安装步骤

#### 1. 准备工作

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 更新后重启（可选）
sudo reboot
```

#### 2. 安装 Xray

```bash
# 使用官方安装脚本
bash <(curl -L https://github.com/XTLS/Xray-install/raw/main/install-release.sh) install

# 验证安装
xray version
```

#### 3. 生成 REALITY 密钥（关键）

```bash
# 生成密钥对
/usr/local/bin/xray x25519

# 输出示例：
# PrivateKey: XXXXXXXX (私钥，用于服务端)
# Password:   XXXXXXXXXX (公钥，用于客户端)
```

#### 4. 生成 UUID

```bash
cat /proc/sys/kernel/random/uuid
```

#### 5. 配置 Xray

```bash
# 编辑配置文件
sudo nano /usr/local/etc/xray/config.json
```

**完整配置示例：**

```json
{
  "log": {
    "loglevel": "warning"
  },
  "inbounds": [
    {
      "listen": "0.0.0.0",
      "port": 443,
      "protocol": "vless",
      "settings": {
        "clients": [
          {
            "id": "你的UUID",
            "flow": "xtls-rprx-vision"
          }
        ],
        "decryption": "none"
      },
      "streamSettings": {
        "network": "tcp",
        "security": "reality",
        "realitySettings": {
          "dest": "www.cloudflare.com:443",
          "serverNames": ["www.cloudflare.com"],
          "privateKey": "你的PrivateKey",
          "shortIds": [""]
        }
      }
    }
  ],
  "outbounds": [
    {
      "protocol": "freedom"
    }
  ]
}
```

**常用伪装网站推荐：**
- `www.cloudflare.com`
- `www.apple.com`
- `www.microsoft.com`
- `time.com`
- `www.yahoo.com`

#### 6. 验证并启动

```bash
# 验证配置文件（无输出=正确）
sudo xray run -test -config /usr/local/etc/xray/config.json

# 启动服务
sudo systemctl start xray

# 设置开机自启
sudo systemctl enable xray

# 检查服务状态
sudo systemctl status xray

# 确认端口监听
ss -lntp | grep 443

# 查看日志
tail -f /var/log/xray/access.log
```

#### 7. 配置防火墙

```bash
# Ubuntu/Debian
sudo ufw allow 22    # SSH
sudo ufw allow 443   # Xray
sudo ufw enable

# CentOS
sudo firewall-cmd --add-port=443/tcp --permanent
sudo firewall-cmd --reload
```

### 客户端配置

#### Clash Meta 配置（YAML）

```yaml
proxies:
  - name: VLESS-Reality
    type: vless
    server: 你的服务器IP
    port: 443
    uuid: 你的UUID
    network: tcp
    tls: true
    udp: true
    flow: xtls-rprx-vision
    servername: www.cloudflare.com
    reality-opts:
      public-key: 你的Password
      short-id: ""
    skip-cert-verify: false

proxy-groups:
  - name: Proxy
    type: select
    proxies:
      - VLESS-Reality

  - name: Auto
    type: url-test
    proxies:
      - VLESS-Reality
    url: 'http://www.gstatic.com/generate_204'
    interval: 300

rules:
  # 直连
  - DOMAIN-SUFFIX,cn,DIRECT
  - DOMAIN-KEYWORD,baidu,DIRECT
  - GEOIP,CN,DIRECT

  # 代理
  - MATCH,Proxy
```

#### 原始 VLESS 节点链接

```
vless://UUID@服务器IP:443?encryption=none&flow=xtls-rprx-vision&security=reality&sni=www.cloudflare.com&fp=chrome&pbk=你的Password&sid=&type=tcp#节点名称
```

---

## Hysteria2

### 技术特点

| 特性 | 说明 |
|------|------|
| **协议基础** | 基于 QUIC（UDP） |
| **传输效率** | 多路复用、0RTT握手 |
| **速度** | 可达2Gbps吞吐量 |
| **局限性** | UDP易被QoS限制，敏感期抗封锁弱 |

### 适用场景

✅ **适合场景：**
- 常规网络环境
- 追求极速传输
- 弱网环境优化

❌ **不适合场景：**
- 敏感时期
- 高管控环境
- 运营商严格限速UDP

### 一键安装

```bash
# 下载安装脚本
wget -O phy2.sh https://gitcode.com/gh_mirrors/hy/hysteria2/raw/main/phy2.sh
chmod +x phy2.sh
bash phy2.sh

# 运行主程序
wget -O hy2.py https://gitcode.com/gh_mirrors/hy/hysteria2/raw/main/hysteria2.py
chmod +x hy2.py
python3 hy2.py
```

### Clash Meta 配置

```yaml
proxies:
  - name: Hysteria2
    type: hysteria2
    server: 你的服务器IP或域名
    port: 443
    password: 你的密码
    sni: 你的服务器IP或域名
    skip-cert-verify: false
```

---

## 客户端工具

### 全平台对比

| 平台 | 推荐工具 | 协议支持 | 状态 |
|------|----------|----------|------|
| **Windows** | Clash Verge Rev | 全协议 | ✅ 持续更新 |
| **macOS** | Clash Verge Rev / FlClash | 全协议 | ✅ 持续更新 |
| **Linux** | Clash Verge / FlClash | 全协议 | ✅ 持续更新 |
| **Android** | Clash Meta for Android | 全协议 | ✅ 稳定 |
| **iOS** | Shadowrocket | 全协议 | ✅ 稳定（$2.99） |

### Clash Verge Rev 特性

```
┌─────────────────────────────────────────┐
│          Clash Verge Rev 功能          │
├─────────────────────────────────────────┤
│  ✅ 支持 VLESS + REALITY                │
│  ✅ 支持 Hysteria2                      │
│  ✅ TUN模式（全局代理）                 │
│  ✅ 规则模式 / 全局模式                 │
│  ✅ 订阅管理                            │
│  ✅ 实时测速                            │
│  ✅ 严格路由（防泄漏）                  │
└─────────────────────────────────────────┘
```

### 模式说明

| 模式 | 作用 | 覆盖范围 | 适用场景 |
|------|------|----------|----------|
| **系统代理** | 设置HTTP/SOCKS5代理 | 仅支持代理的软件 | 网页浏览、开发调试 |
| **TUN模式** | 创建虚拟网卡 | 拦截所有流量 | 游戏、全局代理 |

---

## 常见问题

### 连接测试

```bash
# 测试端口是否开放
telnet 服务器IP 443

# 检查服务状态
sudo systemctl status xray
sudo systemctl status hysteria-server

# 查看实时日志
journalctl -u xray -f
journalctl -u hysteria-server.service -f
```

### 安全建议

| 建议 | 说明 |
|------|------|
| 🔒 使用强密码/UUID | 避免被暴力破解 |
| 🔒 开启 fail2ban | 防止SSH爆破 |
| 🔒 定期更新 | 修复已知漏洞 |
| 🔒 避免默认端口 | 使用非标准端口降低被扫描风险 |
| 🔒 防火墙配置 | 只开放必要端口 |

### 隐私泄漏检测

| 检测项 | 工具 | 预期结果 |
|--------|------|----------|
| IP泄漏 | [ip.sb](https://ip.sb) | 显示VPS的IP |
| DNS泄漏 | [dnsleaktest.com](https://dnsleaktest.com) | 无DNS泄漏 |
| WebRTC泄漏 | [browserleaks.com/webrtc](https://browserleaks.com/webrtc) | No Leak |

### 泄漏解决方案

```bash
# 1. 启用 TUN 模式
Clash Verge 设置 → TUN 模式 → 启用

# 2. 开启严格路由
Clash Verge 设置 → TUN 模式 → 严格路由

# 3. 使用全局模式
Clash Verge 设置 → 代理模式 → 全局

# 4. 禁用 IPv6（可选）
# Windows: 控制面板 → 网络适配器 → 属性 → 取消IPv6
```

---

## 学习资源

- [Xray 官方文档](https://xtls.github.io/)
- [Hysteria2 GitHub](https://github.com/apernet/hysteria2)
- [Clash Meta 文档](https://wiki.metacubex.one/)

---

## 相关笔记

- [[随手记/科学上网/00-索引]]

---

> [!summary] 标签
> #VPS #代理 #网络 #技术学习
