# 守望者（Vigil）— 有机农场在地 AI 巡检机器人

> 黑客松项目 · 2026.05.22 - 05.24

通过 AI 摄像头机器人自动巡视有机农场，拍摄蔬菜生长、田间劳作、昆虫生态等场景，自动剪辑为巡视视频、生成可视化网页报告，并通过微信推送给订阅用户，建立生产者与消费者之间的信任。

---

## 仓库结构

```
.
├── xiaolv/                          # [功能1] 在地巡检 AI 机器人「小绿」
│   ├── xiaolv.py                    #   完整版（761行）：PTZ控制 + 语音 + AI对话 + 发微信
│   └── run.py                       #   精简版（272行）：语音唤醒 → 拍照 → 发微信
│
├── farm-order-hub/                  # [功能2] 农场订单管理系统
│   ├── app.py                       #   Flask Web 主程序 + REST API
│   ├── farm_order_hub/
│   │   ├── models.py                #   数据模型（Order / Event / Rule）
│   │   ├── storage.py               #   SQLite 存储层
│   │   ├── excel_io.py              #   Excel 导入导出
│   │   ├── rules.py                 #   事件规则引擎
│   │   ├── service.py               #   业务服务层
│   │   └── batch.py                 #   批次管理
│   ├── templates/                   #   5个 Jinja2 页面模板
│   └── example.py                   #   使用示例
│
├── web-report/                      # [功能3] 农场巡视报告网页
│   └── 农场巡视报告.html             #   517KB 单文件，60+张照片 base64 内嵌，直接打开即可
│
├── xiaozhi-server/                  # [功能4] 小智语音服务端配置
│   ├── xiaozhi-config.yaml          #   服务端配置（ASR/LLM/TTS/VAD 模块选择）
│   ├── start_xiaozhi.sh             #   启动脚本
│   └── sessions/                    #   语音对话会话记录（3个 .jsonl.gz）
│
├── demo/                            # 演示视频
│   ├── 守望者_演示.mp4               #   项目整体演示（2.6MB）
│   └── 农场巡视报告_final.mp4        #   AI剪辑的巡视报告视频（11MB）
│
└── 素材/昆虫图鉴/                    # 从昆虫图鉴PDF提取的10张高清图片
```

---

## 功能说明

### 功能1：小绿巡检机器人 (`xiaolv/`)

语音唤醒的农场巡检 AI 机器人，能自主找人、对话、拍照并通过微信发送。

**硬件：** Insta360 Link 2 摄像头（PTZ云台） + MacBook 麦克风 + Android 手机（ADB）

**工作流程：**
```
待机（摄像头归中）→ 语音唤醒"小绿" → PTZ蛇形扫描找人（OpenCV人脸检测）
→ AI追踪锁定 → 语音打招呼 → 对话循环（语音→拍照→AI多模态理解→TTS播报）
→ 用户说"拍照发微信" → ADB推送照片到手机 → VLM识别微信按钮 → 自动发送
```

**核心技术：**

| 模块 | 技术方案 | 代码位置 |
|------|----------|----------|
| PTZ 云台控制 | uvc-util 命令行 → Insta360 UVC 协议 | `PTZController` 类 |
| 蛇形扫描找人 | 5×18 步进扫描 + OpenCV DNN (SSD+ResNet) 人脸检测 | `PersonFinder` 类 |
| 语音识别 | SenseVoice 离线 ASR + Silero VAD，自动重采样 | `VoiceEngine` 类 |
| AI 对话 | Gemini 3.5 Flash 多模态（文本+实时画面） | `AIChat` 类 |
| 微信自动发送 | ADB + MiniCPM-V 本地 VLM 识别按钮坐标 | `WeChatSender` 类 |

### 功能2：农场订单管理系统 (`farm-order-hub/`)

为有机农场配送业务提供数据处理后台，处理衢州蔬菜组的报菜订单和售后事件。

- **Excel 导入**：解析客服录入表（订单号/事件类型/客户信息），支持新增、中断、恢复、改地址、取消
- **规则引擎**：按事件类型自动分派处理，支持自定义规则注册
- **Excel 导出**：自动生成发货单
- **Web 界面**：Flask + Jinja2，5 个页面 + REST API

### 功能3：农场巡视报告网页 (`web-report/`)

模拟"守望者机器人视角"的每日巡视报告，单 HTML 文件直接打开。

- 开机动画（模拟机器人启动）→ HUD 状态栏 → 时间线卡片流 → 昆虫科普 → 打包发货 → 日落片尾
- 深色 HUD 风格（`#0f0f1e` + `#00ff88`），CSS 动画，移动端适配
- 60+ 张农场照片 base64 内嵌，无需服务器

### 功能4：自动剪辑巡视视频 (`demo/农场巡视报告_final.mp4`)

用 Claude Code 生成分镜脚本和旁白文案，通过剪映合成，经 5 轮迭代打磨。

50 秒短视频覆盖：晨雾田野 → 蔬菜特写 → 虫害观察 → 采摘劳作 → 昆虫科普 → 打包发货 → 日落收尾

### 功能5：小智语音服务端 (`xiaozhi-server/`)

基于 xiaozhi-server 框架的语音助手服务端配置，集成 SileroVAD + SherpaASR + Gemini LLM + EdgeTTS。

---

## 技术栈

| 类别 | 技术 |
|------|------|
| AI 对话 | Gemini 3.5 Flash（多模态）、MiniCPM-V（本地 VLM） |
| 语音 | SenseVoice ASR、Silero VAD、Edge TTS（全部离线/本地） |
| 视觉 | OpenCV DNN（人脸检测）、Insta360 Link 2（PTZ 控制） |
| 后端 | Python、Flask、SQLite、openpyxl |
| 前端 | HTML/CSS/JS（HUD 风格单页应用） |
| 自动化 | ADB（Android 手机控制）、uvc-util（摄像头 UVC 协议） |
| AI 开发工具 | Claude Code（代码开发 + 分镜脚本 + 网页生成） |
| 视频制作 | 剪映 JianyingPro |

---

## 完成时间线

| 日期 | 工作内容 |
|------|----------|
| 5/22 周五 | 订单管理系统全栈开发，导入衢州蔬菜订单数据测试 |
| 5/23 周六 | 收集 60+ 张农场照片和 5 段视频；提取昆虫图鉴；AI 生成分镜脚本；剪辑视频（5 轮迭代）；生成巡视报告网页 |
| 5/24 周日 | 开发小绿巡检机器人；调通 PTZ 控制、语音识别、AI 对话、微信发送全链路 |
