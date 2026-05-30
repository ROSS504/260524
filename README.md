# 守望者（Vigil）— 项目技术实现文档

> 有机农场在地 AI 巡检机器人 · 黑客松提交

---

## 项目总览

守望者是一套有机农场在地 AI 巡检系统，包含四个已实现的功能模块：

| # | 功能 | 代码位置 | 行数 |
|---|------|----------|------|
| 1 | 在地巡检机器人：摄像头PTZ控制 + 语音交互 + 拍照自动发微信 | `~/xiaolv/xiaolv.py` | 761行 |
| 2 | 巡检机器人精简版（语音唤醒→拍照→发微信） | `~/xiaolv/run.py` | 272行 |
| 3 | 农场巡视报告网页（素材→可视化网页） | `~/Desktop/黑客松/农场巡视报告.html` | 517KB |
| 4 | 农场订单管理系统（Excel导入→规则处理→发货单） | `~/farm-order-hub/` | 8个源文件 |

另有自动剪辑的巡视视频 `农场巡视报告_final.mp4`，通过 Claude Code 生成分镜脚本 + 剪映合成。

---

## 功能一：在地巡检机器人（小绿）

### 1.1 系统架构

```
语音唤醒 "小绿"
    ↓
PTZ蛇形扫描找人（OpenCV人脸检测）
    ↓
AI追踪锁定 → 语音打招呼
    ↓
对话循环（语音识别 → 拍照 → AI多模态理解 → TTS播报）
    ↓
执行任务：拍照 → AI生成描述 → ADB推送照片 → 自动操作微信发送
    ↓
回到待机
```

### 1.2 硬件

- 摄像头：Insta360 Link 2（PTZ云台，通过 uvc-util 控制 UVC 协议）
- 麦克风：MacBook Pro 内置
- 手机：Android（ADB 无线连接，用于操作微信）

### 1.3 源码：`~/xiaolv/xiaolv.py`

```python
#!/usr/bin/env python3
"""
小绿 - 视觉语音AI助手
流程: 待机(归中) → "小绿"唤醒 → 扫描找人+追踪 → 语音反馈 → 对话循环 → 执行任务(拍照/发微信)
"""
import os, sys, time, re, json, signal, subprocess, base64, asyncio
import numpy as np
import pyaudio
import cv2
import sherpa_onnx
import edge_tts
import openai
import requests

# ============================================================
# 配置
# ============================================================
WAKE_WORDS = ["小绿", "小律", "小旅", "小吕", "小驴", "小率", "小路",
              "小录", "小陆", "小六", "小鹿", "绿", "律", "考律", "哨绿",
              "小虑", "小滤", "小侣", "小履"]
WAKE_WORD_DISPLAY = "小绿"

CAMERA_INDEX = 0          # Insta360 Link 2 (OpenCV index)
MIC_DEVICE_INDEX = 2      # MacBook Pro麦克风
SAMPLE_RATE = 16000
CHUNK = 512

API_KEY = "sk-***"
API_BASE = "https://ai.sendercloud.net/v1"
API_MODEL = "gemini-3.5-flash"
TTS_VOICE = "zh-CN-YunxiNeural"

VLM_URL = "http://localhost:11434"
VLM_MODEL = "minicpm-v"

UVC_UTIL = "/tmp/uvc-util"

MODELS_DIR = os.path.expanduser("~/xiaozhi-server/main/xiaozhi-server/models")
ASR_MODEL = os.path.join(MODELS_DIR, "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx")
ASR_TOKENS = os.path.join(MODELS_DIR, "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt")
VAD_MODEL = os.path.join(MODELS_DIR, "snakers4_silero-vad/src/silero_vad/data/silero_vad.onnx")
FACE_PROTO = "/tmp/deploy.prototxt"
FACE_MODEL = "/tmp/res10_300x300_ssd_iter_140000.caffemodel"

DEFAULT_CONTACT = "拾捌"

SYSTEM_PROMPT = f"""你是{WAKE_WORD_DISPLAY}，一个友好的AI助手，通过摄像头和麦克风与用户面对面交流。
你能看到用户发来的照片，也能听到他们说的话。
回复要简短自然，像朋友聊天一样。每次回复控制在1-2句话。

当用户要求拍照/发微信/发照片等，在回复末尾加标签：
[ACTION:photo_wechat]
只在用户明确要求时才加。普通聊天不加任何标签。"""


# ============================================================
# PTZ 摄像头控制
# ============================================================
class PTZController:
    """通过 uvc-util 命令行工具控制 Insta360 Link 2 的云台"""
    def __init__(self):
        self.util = UVC_UTIL
        self.available = os.path.exists(self.util)

    def _run(self, *args):
        if not self.available:
            return ""
        try:
            r = subprocess.run([self.util] + [str(a) for a in args],
                               capture_output=True, text=True, timeout=5)
            return r.stdout.strip()
        except Exception as e:
            print(f"  PTZ error: {e}", flush=True)
            return ""

    def status(self):
        out = self._run("status")
        m = re.search(r'Pan=(-?\d+)\s+Tilt=(-?\d+)\s+Zoom=(\d+)', out)
        return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (0, 0, 100)

    def center(self):
        self._run("center")

    def track_on(self):
        self._run("frame", 2)
        time.sleep(0.3)
        self._run("track_on")

    def track_off(self):
        self._run("track_off")

    def pantilt(self, pan, tilt):
        self._run("pantilt", pan, tilt)


# ============================================================
# 摄像头
# ============================================================
class Camera:
    def __init__(self, index=CAMERA_INDEX):
        self.index = index
        self._cap = None

    def open(self):
        if self._cap and self._cap.isOpened():
            return
        self._cap = cv2.VideoCapture(self.index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        for _ in range(10):
            self._cap.read()
        print("  摄像头已打开", flush=True)

    def capture(self, path="/tmp/xiaolv_photo.jpg"):
        self.open()
        for _ in range(5):
            self._cap.read()
        ret, frame = self._cap.read()
        if ret:
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return path
        return None

    def read_frame(self):
        self.open()
        ret, frame = self._cap.read()
        return frame if ret else None

    def flush(self, n=5):
        if self._cap and self._cap.isOpened():
            for _ in range(n):
                self._cap.read()

    def release(self):
        if self._cap:
            self._cap.release()
            self._cap = None


# ============================================================
# 人脸扫描找人
# ============================================================
class PersonFinder:
    """PTZ蛇形扫描 + OpenCV DNN人脸检测，找到后启动AI追踪"""
    def __init__(self, ptz: PTZController, camera: Camera):
        self.ptz = ptz
        self.camera = camera
        self.net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

    def _detect_face(self, frame):
        fh, fw = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                      1.0, (300, 300), (104, 177, 123))
        self.net.setInput(blob)
        dets = self.net.forward()
        for i in range(dets.shape[2]):
            conf = dets[0, 0, i, 2]
            if conf > 0.7:
                box = dets[0, 0, i, 3:7] * [fw, fh, fw, fh]
                x1, y1, x2, y2 = box.astype(int)
                if (x2 - x1) >= 60 and (y2 - y1) >= 60:
                    return conf, (x1, y1, x2, y2)
        return None, None

    def scan_and_find(self):
        """蛇形扫描找人，找到后开启AI追踪"""
        print("  扫描找人...", flush=True)
        self.ptz.track_on()
        self.camera.open()
        time.sleep(2)
        self.camera.flush(20)
        self.ptz.track_off()

        tilt_levels = [0, 50000, -50000, 100000, 150000]
        pan_range = list(range(-400000, 500000, 50000))

        for ti, tilt_v in enumerate(tilt_levels):
            pans = pan_range if ti % 2 == 0 else list(reversed(pan_range))
            self.ptz.pantilt(pans[0], tilt_v)
            time.sleep(0.5)
            self.camera.flush(10)

            for pan_v in pans:
                self.ptz.pantilt(pan_v, tilt_v)
                time.sleep(0.25)
                self.camera.flush(5)

                confirmed = 0
                for _ in range(6):
                    frame = self.camera.read_frame()
                    if frame is None:
                        continue
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if cv2.Laplacian(gray, cv2.CV_64F).var() < 50:
                        continue
                    conf, box = self._detect_face(frame)
                    if conf:
                        confirmed += 1
                        if confirmed >= 2:
                            print(f"  找到人! pan={pan_v} tilt={tilt_v} conf={conf:.0%}", flush=True)
                            time.sleep(0.5)
                            self.ptz.track_on()
                            return True

            sys.stdout.write(".")
            sys.stdout.flush()

        print("\n  未找到人，归中", flush=True)
        self.ptz.center()
        return False


# ============================================================
# 语音模块 (ASR + VAD + TTS)
# ============================================================
class VoiceEngine:
    """SenseVoice离线ASR + Silero VAD + Edge TTS"""
    def __init__(self):
        print("  加载ASR模型...", flush=True)
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=ASR_MODEL, tokens=ASR_TOKENS,
            num_threads=2, language="zh", use_itn=True,
        )
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = VAD_MODEL
        config.silero_vad.threshold = 0.2
        config.silero_vad.min_silence_duration = 0.8
        config.silero_vad.min_speech_duration = 0.2
        config.sample_rate = SAMPLE_RATE
        self.vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=60)
        self.pa = pyaudio.PyAudio()
        self.stream = None

    def open_mic(self):
        if self.stream:
            return
        dev_info = self.pa.get_device_info_by_index(MIC_DEVICE_INDEX)
        native_rate = int(dev_info['defaultSampleRate'])
        self.native_rate = native_rate
        self.need_resample = (native_rate != SAMPLE_RATE)
        if self.need_resample:
            self.stream = self.pa.open(
                format=pyaudio.paFloat32, channels=1, rate=native_rate,
                input=True, input_device_index=MIC_DEVICE_INDEX,
                frames_per_buffer=int(CHUNK * native_rate / SAMPLE_RATE))
        else:
            self.stream = self.pa.open(
                format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE,
                input=True, input_device_index=MIC_DEVICE_INDEX,
                frames_per_buffer=CHUNK)

    def _resample(self, samples, from_rate, to_rate):
        if from_rate == to_rate:
            return samples
        ratio = to_rate / from_rate
        n_out = int(len(samples) * ratio)
        indices = np.arange(n_out) / ratio
        indices = np.clip(indices, 0, len(samples) - 1)
        return np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)

    def listen(self, timeout=30):
        """监听一句话，返回识别文本。超时返回None"""
        self.open_mic()
        is_speaking = False
        speech_samples = []
        last_activity = time.time()

        while True:
            try:
                chunk_size = int(CHUNK * self.native_rate / SAMPLE_RATE) if self.need_resample else CHUNK
                data = self.stream.read(chunk_size, exception_on_overflow=False)
            except Exception:
                time.sleep(0.01)
                continue

            samples = np.frombuffer(data, dtype=np.float32)
            if self.need_resample:
                samples = self._resample(samples, self.native_rate, SAMPLE_RATE)
            self.vad.accept_waveform(samples)

            if self.vad.is_speech_detected():
                last_activity = time.time()
                if not is_speaking:
                    is_speaking = True
                    speech_samples = []
                speech_samples.append(samples.copy())
                if len(speech_samples) > 300:
                    break
            elif is_speaking:
                break
            else:
                if time.time() - last_activity > timeout:
                    return None

        if not speech_samples:
            return None
        all_audio = np.concatenate(speech_samples)
        s = self.recognizer.create_stream()
        s.accept_waveform(SAMPLE_RATE, all_audio)
        self.recognizer.decode_stream(s)
        text = s.result.text.strip()
        return text if len(text) >= 2 else None

    async def speak(self, text):
        tmp = "/tmp/xiaolv_tts.mp3"
        try:
            comm = edge_tts.Communicate(text, TTS_VOICE, rate="+15%")
            await comm.save(tmp)
            subprocess.run(["afplay", tmp], capture_output=True, timeout=30)
        except Exception:
            subprocess.run(["say", "-v", "Ting-Ting", text],
                           capture_output=True, timeout=30)

    def speak_sync(self, text):
        asyncio.run(self.speak(text))

    def flush_mic(self):
        if not self.stream:
            return
        for _ in range(30):
            try:
                chunk_size = int(CHUNK * self.native_rate / SAMPLE_RATE) if self.need_resample else CHUNK
                data = self.stream.read(chunk_size, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.float32)
                if self.need_resample:
                    samples = self._resample(samples, self.native_rate, SAMPLE_RATE)
                self.vad.accept_waveform(samples)
            except:
                pass

    def play_tone(self):
        subprocess.run(["afplay", "/System/Library/Sounds/Tink.aiff"],
                       capture_output=True, timeout=5)

    def close_mic(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def cleanup(self):
        self.close_mic()
        self.pa.terminate()


# ============================================================
# AI 对话（多模态：文本+图片）
# ============================================================
class AIChat:
    def __init__(self):
        self.client = openai.OpenAI(api_key=API_KEY, base_url=API_BASE)
        self.history = []

    def ask(self, text, image_path=None):
        content = []
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        content.append({"type": "text", "text": text})
        self.history.append({"role": "user", "content": content})

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history[-16:]
        try:
            resp = self.client.chat.completions.create(
                model=API_MODEL, max_tokens=500, messages=messages)
            reply = resp.choices[0].message.content or "..."
            self.history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            return "抱歉，出了点问题。"

    def reset(self):
        self.history.clear()


# ============================================================
# VLM 手机屏幕识别（本地 Ollama + MiniCPM-V）
# ============================================================
class PhoneVLM:
    def __init__(self):
        self.url = VLM_URL
        self.model = VLM_MODEL

    def available(self):
        try:
            r = requests.get(f"{self.url}/api/tags", timeout=3)
            models = [m["name"] for m in r.json().get("models", [])]
            return any(self.model in m for m in models)
        except:
            return False

    def ask(self, image_path, prompt):
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        try:
            r = requests.post(f"{self.url}/api/generate", json={
                "model": self.model,
                "prompt": prompt,
                "images": [b64],
                "stream": False,
            }, timeout=60)
            return r.json().get("response", "")
        except Exception as e:
            return ""


# ============================================================
# 微信发送 (ADB + VLM)
# ============================================================
class WeChatSender:
    """通过ADB控制Android手机操作微信，VLM识别按钮位置"""
    def __init__(self, vlm: PhoneVLM):
        self.vlm = vlm
        self._fix_adb()

    def _fix_adb(self):
        r = subprocess.run("adb devices", shell=True, capture_output=True, text=True, timeout=5)
        if r.stdout.count('\tdevice') > 1:
            for line in r.stdout.split('\n'):
                if 'adb-' in line and '\tdevice' in line:
                    dev = line.split('\t')[0]
                    subprocess.run(f"adb disconnect {dev}", shell=True,
                                   capture_output=True, timeout=5)

    def adb(self, cmd):
        r = subprocess.run(f"adb {cmd}", shell=True, capture_output=True,
                           text=True, timeout=15)
        if 'more than one device' in r.stderr:
            self._fix_adb()
            r = subprocess.run(f"adb {cmd}", shell=True, capture_output=True,
                               text=True, timeout=15)
        return r.stdout.strip()

    def screenshot(self, local_path="/tmp/phone_screen.png"):
        self.adb("shell screencap -p /data/local/tmp/screen.png")
        self.adb(f"pull /data/local/tmp/screen.png {local_path}")
        return local_path

    def tap(self, x, y):
        self.adb(f"shell input tap {x} {y}")
        time.sleep(0.5)

    def _vlm_find(self, prompt):
        """截图 → VLM识别 → 返回坐标"""
        screen = self.screenshot()
        answer = self.vlm.ask(screen, prompt)
        m = re.search(r'(\d{2,4})\s*[,，]\s*(\d{2,4})', answer)
        if m:
            return int(m.group(1)), int(m.group(2))
        return None, None

    def send_photo(self, photo_path, message=None):
        """发送照片到微信，VLM识别按钮 + 固定坐标双保险"""
        print("  微信: 开始发送...", flush=True)
        use_vlm = self.vlm.available()

        # 1. 推送照片到手机相册
        remote = "/sdcard/DCIM/Camera/xiaolv_photo.jpg"
        self.adb(f"shell rm -f {remote}")
        time.sleep(0.3)
        self.adb(f"push {photo_path} {remote}")
        self.adb(f"shell am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE "
                 f"-d 'file://{remote}'")
        time.sleep(3)

        # 2. 确保微信在前台
        self.adb("shell input keyevent KEYCODE_BACK")
        time.sleep(0.3)
        self.adb("shell am start -n com.tencent.mm/.ui.LauncherUI")
        time.sleep(1.5)

        if use_vlm:
            success = self._send_with_vlm(message)
        else:
            success = self._send_with_coords(message)

        self.adb(f"shell rm -f {remote}")
        return success

    def _send_with_vlm(self, message=None):
        """VLM引导：截图→识别按钮位置→点击"""
        # 点+号
        x, y = self._vlm_find(
            "这是微信聊天页面的截图。请找到右下角的加号(+)按钮的中心坐标。只返回坐标，格式: x, y")
        self.tap(x or 680, y or 1570)
        time.sleep(1.5)

        # 点相册
        x, y = self._vlm_find(
            "这是微信的功能面板截图。请找到'相册'图标的中心坐标。只返回坐标，格式: x, y")
        self.tap(x or 110, y or 1170)
        time.sleep(3)

        # 选第一张照片
        x, y = self._vlm_find(
            "这是手机相册选择页面。请找到左上角第一张照片的中心坐标。只返回坐标，格式: x, y")
        self.tap(x or 150, y or 166)
        time.sleep(1)

        # 点发送
        x, y = self._vlm_find(
            "这是照片选择页面，下方有发送按钮。请找到'发送'按钮的中心坐标。只返回坐标，格式: x, y")
        self.tap(x or 637, y or 1565)
        time.sleep(2)

        if message:
            self._send_text(message)
        print("  微信: 完成!", flush=True)
        return True

    def _send_with_coords(self, message=None):
        """固定坐标fallback"""
        self.tap(680, 1570)   # +号
        time.sleep(1.5)
        self.tap(110, 1170)   # 相册
        time.sleep(3)
        self.tap(150, 166)    # 第一张照片
        time.sleep(1)
        self.tap(637, 1565)   # 发送
        time.sleep(2)
        if message:
            self._send_text(message)
        print("  微信: 完成!", flush=True)
        return True

    def _send_text(self, message):
        """通过AdbKeyboard输入法发送中文文字"""
        self.tap(35, 1570)
        time.sleep(0.3)
        self.tap(35, 1570)
        time.sleep(0.3)
        self.adb("shell ime set com.android.adbkeyboard/.AdbIME")
        time.sleep(0.3)
        self.tap(250, 1140)
        time.sleep(0.3)
        safe_msg = message.replace("'", "'\"'\"'")
        self.adb(f"shell am broadcast -a ADB_INPUT_TEXT --es msg '{safe_msg}'")
        time.sleep(1)
        self.tap(657, 1570)
        time.sleep(1)
        self.adb("shell input keyevent KEYCODE_BACK")
        time.sleep(0.3)
        self.adb("shell ime set com.baidu.input_mi/.ImeService")


# ============================================================
# 主系统
# ============================================================
class XiaoLv:
    def __init__(self):
        self.voice = VoiceEngine()
        self.camera = Camera()
        self.ptz = PTZController()
        self.finder = PersonFinder(self.ptz, self.camera)
        self.ai = AIChat()
        self.vlm = PhoneVLM()
        self.wechat = WeChatSender(self.vlm)

    def cleanup(self):
        self.ptz.center()
        self.camera.release()
        self.voice.cleanup()

    def standby(self):
        """待机：摄像头归中，持续监听唤醒词"""
        self.ptz.center()
        self.ptz.track_off()
        self.ai.reset()
        while True:
            text = self.voice.listen(timeout=9999)
            if text:
                if any(w in text for w in WAKE_WORDS):
                    return text

    def activate(self):
        """唤醒后：扫描找人 → 锁定追踪 → 语音反馈"""
        self.voice.play_tone()
        self.voice.speak_sync("已启动，我来找你")
        self.camera.open()
        found = self.finder.scan_and_find()
        if found:
            self.ptz.track_on()
            time.sleep(1)
            photo = self.camera.capture()
            reply = self.ai.ask("你刚刚启动并找到了用户，简短打个招呼。", photo)
            reply = self._strip_actions(reply)
            self.voice.speak_sync(reply)
        else:
            self.voice.speak_sync("没有找到你，但我已经启动了。")
        self.voice.flush_mic()

    def chat_loop(self):
        """持续对话，每轮拍照+AI理解+TTS播报，触发拍照发微信"""
        while True:
            text = self.voice.listen(timeout=120)
            if text is None:
                self.voice.speak_sync("你不说话了，我先休息了。")
                return
            photo = self.camera.capture()
            reply = self.ai.ask(text, photo)
            has_action = "[ACTION:photo_wechat]" in reply
            clean = self._strip_actions(reply)
            self.voice.speak_sync(clean)
            self.voice.flush_mic()
            if has_action:
                self._do_photo_wechat(photo)
            self.ptz.track_on()

    def _strip_actions(self, reply):
        return re.sub(r'\s*\[ACTION:\w+\]', '', reply).strip()

    def _do_photo_wechat(self, photo_path):
        """拍照 → AI生成描述 → 发微信"""
        final_photo = self.camera.capture("/tmp/xiaolv_wechat_photo.jpg")
        if not final_photo:
            self.voice.speak_sync("拍照失败了")
            return
        desc = self.ai.ask("根据这张照片，用一句简短的话描述。", final_photo)
        desc = self._strip_actions(desc)
        msg = f"{WAKE_WORD_DISPLAY}AI拍摄: {desc}"
        self.voice.speak_sync(f"正在发送照片给{DEFAULT_CONTACT}")
        self.voice.flush_mic()
        success = self.wechat.send_photo(final_photo, message=msg)
        if success:
            self.voice.speak_sync(f"照片已发给{DEFAULT_CONTACT}了")
        self.ptz.track_on()
        self.voice.flush_mic()

    def run(self):
        """主循环: 待机 → 唤醒 → 激活 → 对话 → 回到待机"""
        signal.signal(signal.SIGINT, lambda s, f: self.cleanup() or sys.exit(0))
        try:
            while True:
                self.standby()
                self.activate()
                self.chat_loop()
        finally:
            self.cleanup()


if __name__ == "__main__":
    app = XiaoLv()
    app.run()
```

### 1.4 关键技术点说明

| 技术点 | 实现方式 |
|--------|----------|
| PTZ云台控制 | 通过 `uvc-util` 命令行工具向 Insta360 Link 2 发送 UVC 协议指令（pan/tilt/zoom/track_on/track_off/center） |
| 蛇形扫描找人 | 5级俯仰(tilt) × 18级水平(pan)步进扫描，每个位置用 OpenCV DNN (SSD+ResNet) 做人脸检测，连续2帧确认后锁定 |
| 模糊帧过滤 | 用拉普拉斯算子方差 `cv2.Laplacian(gray, cv2.CV_64F).var() < 50` 判断并跳过运动模糊帧 |
| 语音识别 | SenseVoice 离线模型 (sherpa-onnx)，Silero VAD 端点检测，支持设备采样率自动重采样到16kHz |
| 唤醒词容错 | 列举19种"小绿"的同音/近音变体，覆盖ASR常见误识别 |
| AI多模态对话 | 每轮对话将摄像头实时画面(base64) + 用户语音文本一起发给 Gemini 3.5 Flash |
| 微信自动操作 | ADB无线连接Android → 推送照片到相册 → VLM(MiniCPM-V)截图识别按钮坐标 → tap操作，失败回退固定坐标 |
| 中文输入 | 切换到 AdbKeyboard 输入法，通过 broadcast 发送中文文本 |

---

## 功能二：自动剪辑巡视报告视频

### 2.1 素材

- 5段农场实拍视频（0.mp4 – 4.mp4，总计约85MB）
- 60+张微信传回的现场照片（蔬菜、田野、晨雾、打包、日落等）
- 10张从《绿手指无脊椎动物150种昆虫图鉴》PDF中提取的高清昆虫图
- 1段微信视频（鸡鸭散步）、1段大姐干活视频

### 2.2 分镜脚本（由 Claude Code 生成）

```
时间   画面素材                          旁白
────────────────────────────────────────────────────────────
0-3s   片头卡                           "早上好～ 我又来巡田了"
3-8s   151858(晨雾蓝天) + 151815(田野)  "今天衢州天气不错，昨晚下了点雨，菜都精神了"
8-13s  0.mp4 20-25s(西葫芦特写) + HUD   "你看这个西葫芦，昨天还小小的，今天就胖了一圈"
13-16s 151737(叶菜)                     "这片油冬菜长得太快了，再不摘就老了"
16-18s 0.mp4 58-62s(毛毛虫实拍) + HUD   "@汪姐 你的菜又有小客人啦"
18-20s insect_04(蝽在果实)              "有机菜就是这样，大家一起分享嘛"
20-26s 2.mp4 20-26s(采摘)               "大姐们已经开始干活了"
26-32s 3.mp4 8-14s(花菜装篮)            "你上次说不要苦瓜，放心，我盯着呢"
32-36s 微信视频(鸡鸭散步)               "路过鸡窝，看到我也不跑了"
36-39s insect_08(蜾蠃)                  "这是蜾蠃，专门抓害虫，咱菜地的保安"
39-42s insect_10(蟹蛛)                  "这只蟹蛛会变色伪装，比农药好使多了"
42-47s 152125+152129(打包纸箱)          "你的菜已经装箱了，明天顺丰到"
47-50s 151854(日落) → 片尾卡            "今天就到这儿，明天再来给你看看，拜拜～"
```

### 2.3 制作过程

1. 用 Claude Code 分析素材文件名和内容，生成完整分镜脚本和旁白文案
2. 用 Claude Code 从昆虫图鉴 PDF 提取 10 张高清昆虫图片
3. 在剪映（JianyingPro）中按分镜脚本剪辑合成
4. 经过 5 轮迭代优化（v1→v2→v3→v4→final），最终输出 `农场巡视报告_final.mp4`（11MB）

---

## 功能三：农场巡视报告网页

### 3.1 文件

`~/Desktop/黑客松/农场巡视报告.html` — 517KB 单文件，所有图片 base64 内嵌，无需服务器直接打开。

### 3.2 页面结构与效果

```
┌─────────────────────────────────┐
│  [开机动画] 守望者 SVG + 进度条  │  ← 模拟机器人启动
│  > SYSTEM BOOT...               │
│  > CAMERA INIT...               │
│  > GPS LOCK...                  │
├─────────────────────────────────┤
│  守望者                         │
│  今日农场巡视报告                │
│  2026年5月23日 · 衢州基地        │
├─────────────────────────────────┤
│  ● 巡视中 · GPS 29.0°N 118.8°E │  ← HUD状态栏，绿色呼吸灯
├─────────────────────────────────┤
│  [晨雾蓝天] [多云田野]           │  ← 双图卡片
│  06:47 · 蔬菜种植区              │
│  "菜都精神了"                    │
├─────────────────────────────────┤
│  [西葫芦特写]                    │  ← 单图卡片
│  06:52 · A区瓜果                 │
│  "昨天还小小的，今天就胖了一圈"    │
├─────────────────────────────────┤
│  🐛 虫虫播报                     │  ← 昆虫卡片（绿色边框）
│  @汪姐 你的菜又有小客人啦         │
├─────────────────────────────────┤
│  📚 昆虫图鉴                     │  ← 科普卡片
│  蜾蠃 — 专门抓害虫，菜地的保安    │
├─────────────────────────────────┤
│  📦 今日打包                     │  ← 打包卡片（橙色边框）
│  你的菜已经装箱了                 │
├─────────────────────────────────┤
│  [日落照片]                      │
│  今天就到这儿，明天再来看看        │
└─────────────────────────────────┘
```

### 3.3 技术实现

- 深色 HUD 风格（`#0f0f1e` 底色 + `#00ff88` 强调色），模拟机器人视角
- CSS 动画：开机进度条、卡片淡入上浮 `fadeUp`、状态灯闪烁 `blink`
- 移动端优先设计（`max-width: 420px`），适合微信内打开
- 60+ 张农场照片全部转 base64 内嵌，单文件完全自包含
- 三类卡片样式：普通巡视（灰色边框）、昆虫科普（绿色 `.insect`）、打包发货（橙色 `.packing`）

### 3.4 CSS 核心代码（摘录）

```css
/* 开机动画 */
.boot-screen {
  position: fixed; top: 0; left: 0; width: 100%; height: 100vh;
  background: #0f0f1e; display: flex; flex-direction: column;
  align-items: center; justify-content: center; z-index: 100;
}

/* HUD状态栏 */
.hud {
  display: flex; justify-content: space-between;
  padding: 8px 16px; background: rgba(0,255,136,0.05);
  border: 1px solid rgba(0,255,136,0.15); border-radius: 8px;
  font-size: 12px; color: #00ff88; font-family: monospace;
}
.hud .dot { width: 6px; height: 6px; border-radius: 50%;
  background: #00ff88; animation: blink 2s infinite; }

/* 时间线卡片 */
.card {
  background: #1a1a2e; border-radius: 16px; margin-bottom: 16px;
  border: 1px solid #252540;
  animation: fadeUp 0.6s ease-out both;
}
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}

/* 昆虫科普卡 */
.card.insect { border-color: rgba(0,255,136,0.3); }
.card.insect .badge {
  background: rgba(0,255,136,0.15); color: #00ff88;
  font-size: 11px; padding: 2px 8px; border-radius: 10px;
}
```

---

## 功能四：农场订单管理系统

### 4.1 文件结构

```
~/farm-order-hub/
├── app.py                          # Flask Web 主程序 (248行)
├── farm_order_hub/
│   ├── models.py                   # 数据模型：Order/Event/Rule (80行)
│   ├── storage.py                  # SQLite 存储层 (195行)
│   ├── excel_io.py                 # Excel 导入导出 (191行)
│   ├── rules.py                    # 规则引擎 (112行)
│   ├── service.py                  # 服务层 (104行)
│   └── batch.py                    # 批次管理
├── templates/
│   ├── index.html                  # 首页：上传Excel + 数据类型列表
│   ├── type_detail.html            # 类型详情：数据集 + 规则 + 对话
│   ├── dataset_detail.html         # 数据集详情：逐行查看
│   ├── rules.html                  # 规则配置
│   └── batch.html                  # 批次管理
├── data/app.db                     # SQLite 数据库
├── uploads/                        # 已上传：衢州蔬菜组报菜.xlsx、售后表.xlsx
└── output/                         # 生成：发货单_2026-05-23.xlsx、客服录入模板.xlsx
```

### 4.2 数据模型 (`models.py`)

```python
class OrderStatus(str, Enum):
    NEW = "new"
    ACTIVE = "active"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"
    COMPLETED = "completed"

class EventType(str, Enum):
    ORDER_NEW = "order_new"
    ORDER_RESUME = "order_resume"
    ORDER_INTERRUPT = "order_interrupt"
    AFTER_SALE_ADDRESS = "after_sale_address"
    AFTER_SALE_CANCEL = "after_sale_cancel"

@dataclass
class Order:
    order_id: str
    customer_name: str
    address: str
    phone: str
    items: str              # JSON字符串
    status: OrderStatus = OrderStatus.NEW
    batch_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
```

### 4.3 规则引擎 (`rules.py`)

```python
class RuleEngine:
    """按事件类型分派处理逻辑，支持自定义规则注册"""
    def __init__(self, storage: Storage):
        self.storage = storage
        self._handlers: dict[EventType, list[RuleHandler]] = defaultdict(list)
        self._register_defaults()

    def process_batch(self, batch_id: str) -> int:
        return self._process_events(self.storage.get_unprocessed_events(batch_id))

    def _dispatch(self, event: Event):
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            handler(self.storage, event)

# 内置规则：新增订单→创建记录，修改地址→更新地址，取消→标记取消...
def _handle_order_new(storage, event):
    data = json.loads(event.payload)
    order = Order(order_id=event.order_id, customer_name=data.get("customer_name", ""), ...)
    storage.save_order(order)

def _handle_address_change(storage, event):
    data = json.loads(event.payload)
    storage.update_order_address(event.order_id, data.get("new_address", ""))
```

### 4.4 Excel 导入导出 (`excel_io.py`)

```python
# 导入格式（客服录入表）:
# | 订单号 | 事件类型 | 客户姓名 | 电话 | 地址 | 商品信息 | 新地址 | 备注 |

def import_events_from_excel(file_path) -> list[dict]:
    """解析客服录入Excel，返回标准化事件列表"""
    wb = load_workbook(str(file_path), read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    events = []
    for row in rows:
        event_type = EVENT_TYPE_LABELS[str(row[1]).strip()]
        payload = {}
        if event_type == EventType.ORDER_NEW:
            payload = {"customer_name": ..., "address": ..., "phone": ...,
                       "items": _parse_items_text(str(row[5]))}
        events.append({"order_id": ..., "event_type": event_type, "payload": payload})
    return events

def export_orders_to_excel(orders, output_path):
    """生成发货单Excel"""
    wb = Workbook()
    ws = wb.active
    ws.title = "发货订单"
    # 写入表头 + 逐行写入订单数据...
```

### 4.5 Web 界面 (`app.py`)

```python
app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload():
    """上传Excel → 解析表头和数据行 → 存入SQLite"""
    file = request.files.get("file")
    wb = load_workbook(str(save_path), read_only=True)
    headers = [str(h) for h in all_rows[0]]
    dataset_id = storage.create_dataset(type_id, file.filename, len(data_rows), headers)
    storage.add_data_rows(dataset_id, data_rows)

# 还提供 REST API 供 Claude Code 调用：
@app.route("/api/dataset/<int:dataset_id>/rows", methods=["GET"])  # 获取数据行
@app.route("/api/row/<int:row_id>/result", methods=["PUT"])        # 更新处理结果
@app.route("/api/type/<int:type_id>/rules", methods=["POST"])      # 添加规则
```

### 4.6 数据库 schema (`storage.py`)

```sql
CREATE TABLE IF NOT EXISTS data_types (
    type_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
    type_id INTEGER NOT NULL,
    file_name TEXT NOT NULL,
    row_count INTEGER NOT NULL DEFAULT 0,
    raw_headers TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY (type_id) REFERENCES data_types(type_id)
);
CREATE TABLE IF NOT EXISTS data_rows (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL,
    row_data TEXT NOT NULL,
    processed INTEGER NOT NULL DEFAULT 0,
    result TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS rules (
    rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
    type_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1
);
```

---

## 完成时间线

| 日期 | 工作内容 |
|------|----------|
| 5/22 周五 | Farm Order Hub 全栈开发（Flask+SQLite+Excel+规则引擎+Web界面），导入实际衢州蔬菜订单数据测试 |
| 5/23 周六 | 收集60+张农场照片和5段视频素材；提取昆虫图鉴图片；Claude Code生成分镜脚本；剪映剪辑视频（5轮迭代）；Claude Code生成农场巡视报告网页（517KB单文件） |
| 5/24 周日 | 开发小绿巡检机器人（精简版run.py + 完整版xiaolv.py）；调通PTZ控制、语音识别、AI对话、微信ADB发送全链路 |

## AI 工具使用

| 工具 | 用途 |
|------|------|
| Claude Code | 全部代码开发、视频分镜脚本、网页生成、昆虫图片提取 |
| Gemini 3.5 Flash | 小绿运行时的多模态AI对话（理解画面+生成回复） |
| MiniCPM-V (Ollama) | 小绿运行时的手机屏幕按钮识别 |
| SenseVoice (sherpa-onnx) | 离线语音识别 |
| Edge TTS | 语音合成 |
| 剪映 JianyingPro | 视频最终合成 |
