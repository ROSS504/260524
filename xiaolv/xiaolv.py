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

# 摄像头 & 麦克风
CAMERA_INDEX = 0          # Insta360 Link 2 (OpenCV index)
MIC_DEVICE_INDEX = 2      # MacBook Pro麦克风
SAMPLE_RATE = 16000
CHUNK = 512

# AI 对话
API_KEY = "sk-YOUR_API_KEY"
API_BASE = "https://ai.sendercloud.net/v1"
API_MODEL = "gemini-3.5-flash"
TTS_VOICE = "zh-CN-YunxiNeural"

# VLM (本地 ollama)
VLM_URL = "http://localhost:11434"
VLM_MODEL = "minicpm-v"

# PTZ
UVC_UTIL = "/tmp/uvc-util"

# ASR / VAD 模型
MODELS_DIR = os.path.expanduser("~/xiaozhi-server/main/xiaozhi-server/models")
ASR_MODEL = os.path.join(MODELS_DIR, "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx")
ASR_TOKENS = os.path.join(MODELS_DIR, "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt")
VAD_MODEL = os.path.join(MODELS_DIR, "snakers4_silero-vad/src/silero_vad/data/silero_vad.onnx")
FACE_PROTO = "/tmp/deploy.prototxt"
FACE_MODEL = "/tmp/res10_300x300_ssd_iter_140000.caffemodel"

# 默认发送对象
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
        # 清空缓冲
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
        # 先 track_on 唤醒摄像头（Insta360 休眠模式需要这个）
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
                    # 跳过模糊帧
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
        # 检查设备原生采样率
        dev_info = self.pa.get_device_info_by_index(MIC_DEVICE_INDEX)
        native_rate = int(dev_info['defaultSampleRate'])
        self.native_rate = native_rate
        self.need_resample = (native_rate != SAMPLE_RATE)

        if self.need_resample:
            # 用设备原生采样率打开，后续重采样到16000
            self.stream = self.pa.open(
                format=pyaudio.paFloat32, channels=1, rate=native_rate,
                input=True, input_device_index=MIC_DEVICE_INDEX,
                frames_per_buffer=int(CHUNK * native_rate / SAMPLE_RATE))
            print(f"  麦克风已打开 (device={MIC_DEVICE_INDEX}, native={native_rate}Hz, resample→{SAMPLE_RATE}Hz)", flush=True)
        else:
            self.stream = self.pa.open(
                format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE,
                input=True, input_device_index=MIC_DEVICE_INDEX,
                frames_per_buffer=CHUNK)
            print(f"  麦克风已打开 (device={MIC_DEVICE_INDEX}, {SAMPLE_RATE}Hz)", flush=True)

    def close_mic(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def flush_mic(self):
        """清空麦克风和VAD缓冲（TTS播放后调用）"""
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

    def _resample(self, samples, from_rate, to_rate):
        """简单线性重采样"""
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
                if len(speech_samples) > 300:  # ~10s max
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
        """TTS播放"""
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

    def play_tone(self):
        subprocess.run(["afplay", "/System/Library/Sounds/Tink.aiff"],
                       capture_output=True, timeout=5)

    def cleanup(self):
        self.close_mic()
        self.pa.terminate()


# ============================================================
# AI 对话
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
            t0 = time.time()
            resp = self.client.chat.completions.create(
                model=API_MODEL, max_tokens=500, messages=messages)
            reply = resp.choices[0].message.content or "..."
            print(f"  AI ({time.time()-t0:.1f}s): {reply}", flush=True)
            self.history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            print(f"  AI error: {e}", flush=True)
            return "抱歉，出了点问题。"

    def reset(self):
        self.history.clear()


# ============================================================
# VLM 手机屏幕识别
# ============================================================
class PhoneVLM:
    """用本地VLM识别手机屏幕内容，返回操作坐标"""

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
        """发送图片+问题给VLM，返回回答"""
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
            print(f"  VLM error: {e}", flush=True)
            return ""


# ============================================================
# 微信发送 (ADB + VLM)
# ============================================================
class WeChatSender:
    def __init__(self, vlm: PhoneVLM):
        self.vlm = vlm
        self._fix_adb()

    def _fix_adb(self):
        """确保只有一个ADB设备连接"""
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
        # 解析坐标 (x, y)
        m = re.search(r'(\d{2,4})\s*[,，]\s*(\d{2,4})', answer)
        if m:
            return int(m.group(1)), int(m.group(2))
        return None, None

    def send_photo(self, photo_path, message=None):
        """
        发送照片到微信（微信需已打开在拾捌对话框）
        优先用VLM识别按钮位置，失败则回退固定坐标
        """
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

        # 清理
        self.adb(f"shell rm -f {remote}")
        return success

    def _send_with_vlm(self, message=None):
        """VLM引导的发送流程"""
        print("  微信: VLM模式", flush=True)

        # 点+号
        x, y = self._vlm_find(
            "这是微信聊天页面的截图。请找到右下角的加号(+)按钮的中心坐标。"
            "只返回坐标，格式: x, y")
        if x:
            self.tap(x, y)
        else:
            self.tap(680, 1570)  # fallback
        time.sleep(1.5)

        # 点相册
        x, y = self._vlm_find(
            "这是微信的功能面板截图。请找到'相册'图标的中心坐标。"
            "只返回坐标，格式: x, y")
        if x:
            self.tap(x, y)
        else:
            self.tap(110, 1170)  # fallback
        time.sleep(3)

        # 选第一张照片（最新的）
        x, y = self._vlm_find(
            "这是手机相册选择页面。请找到左上角第一张照片(最新照片)的中心坐标。"
            "只返回坐标，格式: x, y")
        if x:
            self.tap(x, y)
        else:
            self.tap(150, 166)  # fallback
        time.sleep(1)

        # 点发送
        x, y = self._vlm_find(
            "这是照片选择页面，下方有发送按钮。请找到'发送'按钮的中心坐标。"
            "只返回坐标，格式: x, y")
        if x:
            self.tap(x, y)
        else:
            self.tap(637, 1565)  # fallback
        time.sleep(2)

        # 发文字
        if message:
            self._send_text(message)

        print("  微信: 完成!", flush=True)
        return True

    def _send_with_coords(self, message=None):
        """固定坐标的发送流程（fallback）"""
        print("  微信: 固定坐标模式", flush=True)

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
        """发送文字消息"""
        print(f"  微信: 发文字 [{message[:30]}...]", flush=True)
        self.tap(35, 1570)
        time.sleep(0.3)
        self.tap(35, 1570)
        time.sleep(0.3)
        self.adb("shell ime set com.android.adbkeyboard/.AdbIME")
        time.sleep(0.3)
        self.tap(250, 1140)
        time.sleep(0.3)
        # 转义单引号
        safe_msg = message.replace("'", "'\"'\"'")
        self.adb(f"shell am broadcast -a ADB_INPUT_TEXT --es msg '{safe_msg}'")
        time.sleep(1)
        self.tap(657, 1570)   # 发送
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
        print("\n清理资源...", flush=True)
        self.ptz.center()
        self.camera.release()
        self.voice.cleanup()

    # ---------- 阶段1: 待机 ----------
    def standby(self):
        """待机：摄像头归中，等待唤醒词"""
        print(f'\n{"="*50}', flush=True)
        print(f'  {WAKE_WORD_DISPLAY} - 待机中', flush=True)
        print(f'  说 "{WAKE_WORD_DISPLAY}" 唤醒我', flush=True)
        print(f'{"="*50}\n', flush=True)

        self.ptz.center()
        self.ptz.track_off()
        self.ai.reset()

        while True:
            text = self.voice.listen(timeout=9999)
            if text:
                print(f"  [{text}]", flush=True)
                if any(w in text for w in WAKE_WORDS):
                    print(f"\n*** 唤醒成功! ***\n", flush=True)
                    return text
                # 非唤醒词，继续等
        return None

    # ---------- 阶段2: 启动+找人 ----------
    def activate(self):
        """唤醒后：扫描找人 → 锁定追踪 → 语音反馈"""
        self.voice.play_tone()
        self.voice.speak_sync("已启动，我来找你")

        self.camera.open()
        found = self.finder.scan_and_find()

        if found:
            self.ptz.track_on()
            time.sleep(1)
            # 拍照打招呼
            photo = self.camera.capture()
            reply = self.ai.ask(
                "你刚刚启动并找到了用户，简短打个招呼，说你已经锁定他了。", photo)
            reply = self._strip_actions(reply)
            self.voice.speak_sync(reply)
        else:
            self.voice.speak_sync("没有找到你，但我已经启动了，你说话我能听到。")

        self.voice.flush_mic()

    # ---------- 阶段3: 对话循环 ----------
    def chat_loop(self):
        """持续对话，直到长时间无人说话"""
        print("\n  进入对话模式\n", flush=True)

        while True:
            text = self.voice.listen(timeout=120)
            if text is None:
                print("  长时间无语音，回到待机", flush=True)
                self.voice.speak_sync("你不说话了，我先休息了。")
                return  # 回到待机

            print(f"\n  你: {text}", flush=True)

            # 拍照（给AI看当前画面）
            photo = self.camera.capture()

            # AI回复
            reply = self.ai.ask(text, photo)

            # 解析动作标签
            has_action = "[ACTION:photo_wechat]" in reply
            clean = self._strip_actions(reply)

            # 播报回复
            self.voice.speak_sync(clean)
            self.voice.flush_mic()

            # 执行动作
            if has_action:
                self._do_photo_wechat(photo)

            # 确保追踪还开着
            self.ptz.track_on()

    def _strip_actions(self, reply):
        return re.sub(r'\s*\[ACTION:\w+\]', '', reply).strip()

    def _do_photo_wechat(self, photo_path):
        """拍照 → 生成描述 → 发微信"""
        print("\n  === 执行: 拍照发微信 ===", flush=True)

        # 拍一张新照片
        final_photo = self.camera.capture("/tmp/xiaolv_wechat_photo.jpg")
        if not final_photo:
            self.voice.speak_sync("拍照失败了")
            return

        # 让AI生成描述
        desc = self.ai.ask(
            "根据这张照片，用一句简短的话描述这个人现在在做什么。只说描述，不加标签。",
            final_photo)
        desc = self._strip_actions(desc)
        msg = f"{WAKE_WORD_DISPLAY}AI拍摄: {desc}" if desc else f"{WAKE_WORD_DISPLAY}AI拍摄"

        self.voice.speak_sync(f"正在发送照片给{DEFAULT_CONTACT}")
        self.voice.flush_mic()

        success = self.wechat.send_photo(final_photo, message=msg)

        if success:
            self.voice.speak_sync(f"照片已发给{DEFAULT_CONTACT}了")
        else:
            self.voice.speak_sync("发送失败了")

        # 重新锁定追踪
        self.ptz.track_on()
        self.voice.flush_mic()

    # ---------- 主循环 ----------
    def run(self):
        """主循环: 待机 → 唤醒 → 激活 → 对话 → 回到待机"""
        signal.signal(signal.SIGINT, lambda s, f: self.cleanup() or sys.exit(0))

        print(f"\n{'#'*50}", flush=True)
        print(f"  {WAKE_WORD_DISPLAY} 视觉语音AI助手", flush=True)
        print(f"  摄像头: Insta360 Link 2 (index={CAMERA_INDEX})", flush=True)
        print(f"  麦克风: device={MIC_DEVICE_INDEX}", flush=True)
        print(f"  VLM: {'可用' if self.vlm.available() else '不可用(使用固定坐标)'}", flush=True)
        print(f"{'#'*50}\n", flush=True)

        try:
            while True:
                self.standby()
                self.activate()
                self.chat_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()


if __name__ == "__main__":
    app = XiaoLv()
    app.run()
