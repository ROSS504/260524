#!/usr/bin/env python3
"""
小绿完整流程：语音唤醒 → 拍照 → 发微信(照片+文字)
前提：微信已打开在拾捌对话框
"""
import os, sys, time, subprocess, base64
import numpy as np
import pyaudio
import cv2
import sherpa_onnx

# ============ 配置 ============
WAKE_WORDS = ["小绿", "小律", "小旅", "小吕", "小驴", "小率", "小路",
              "小录", "小陆", "小六", "小鹿", "绿", "律", "考律", "哨绿",
              "小虑", "小滤", "小侣", "小履"]
PHOTO_KEYWORDS = ["拍照", "发送", "发给", "照片", "拍.*发"]

MIC_DEVICE_INDEX = 2   # MacBook Pro麦克风
SAMPLE_RATE = 16000
CAMERA_INDEX = 0       # Insta360 Link 2
UVC_UTIL = "/tmp/uvc-util"

MODELS_DIR = os.path.expanduser("~/xiaozhi-server/main/xiaozhi-server/models")
ASR_MODEL = os.path.join(MODELS_DIR, "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx")
ASR_TOKENS = os.path.join(MODELS_DIR, "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt")
VAD_MODEL = os.path.join(MODELS_DIR, "snakers4_silero-vad/src/silero_vad/data/silero_vad.onnx")


def ptz(*args):
    subprocess.run([UVC_UTIL] + [str(a) for a in args], capture_output=True, timeout=5)


def say(text):
    print(f"  [语音] {text}", flush=True)
    try:
        subprocess.run(["say", "-v", "Tingting", text], timeout=10)
    except:
        pass


def adb(cmd):
    r = subprocess.run(f"adb {cmd}", shell=True, capture_output=True, text=True, timeout=15)
    return r.stdout.strip()


def resample(samples, from_rate, to_rate):
    if from_rate == to_rate:
        return samples
    n_out = int(len(samples) * to_rate / from_rate)
    indices = np.arange(n_out) * from_rate / to_rate
    return np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)


def listen_once(stream, vad, recognizer, native_rate, timeout=30):
    """监听一句话"""
    is_speaking = False
    speech_samples = []
    last_activity = time.time()
    chunk_size = int(512 * native_rate / SAMPLE_RATE)

    while True:
        try:
            data = stream.read(chunk_size, exception_on_overflow=False)
        except:
            time.sleep(0.01)
            continue
        samples = np.frombuffer(data, dtype=np.float32)
        if native_rate != SAMPLE_RATE:
            samples = resample(samples, native_rate, SAMPLE_RATE)
        vad.accept_waveform(samples)

        if vad.is_speech_detected():
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
    audio = np.concatenate(speech_samples)
    s = recognizer.create_stream()
    s.accept_waveform(SAMPLE_RATE, audio)
    recognizer.decode_stream(s)
    text = s.result.text.strip()
    return text if len(text) >= 2 else None


def flush_mic(stream, vad, native_rate):
    chunk_size = int(512 * native_rate / SAMPLE_RATE)
    for _ in range(30):
        try:
            data = stream.read(chunk_size, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.float32)
            if native_rate != SAMPLE_RATE:
                samples = resample(samples, native_rate, SAMPLE_RATE)
            vad.accept_waveform(samples)
        except:
            pass


def take_photo():
    """唤醒摄像头+拍照"""
    ptz("track_on")
    time.sleep(1)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    for _ in range(20):
        cap.read()
    time.sleep(1)
    ret, frame = cap.read()
    path = "/tmp/xiaolv_photo.jpg"
    if ret:
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"  拍照OK", flush=True)
    else:
        print(f"  拍照失败", flush=True)
        path = None
    cap.release()
    return path


def send_wechat(photo_path):
    """发微信：只发照片"""
    print("  微信: 开始发送...", flush=True)

    # 推送照片
    print("  微信: 推送照片...", flush=True)
    adb("shell rm -f /sdcard/DCIM/Camera/xiaolv_photo.jpg")
    adb(f"push {photo_path} /sdcard/DCIM/Camera/xiaolv_photo.jpg")
    adb("shell am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d 'file:///sdcard/DCIM/Camera/xiaolv_photo.jpg'")
    time.sleep(3)

    print("  微信: +号...", flush=True)
    adb("shell input tap 657 1549")
    time.sleep(2)

    print("  微信: 相册...", flush=True)
    adb("shell input tap 108 1170")
    time.sleep(3)

    print("  微信: 选照片...", flush=True)
    adb("shell input tap 90 150")
    time.sleep(1)

    print("  微信: 发送照片...", flush=True)
    adb("shell input tap 621 1573")
    time.sleep(2)

    # 清理
    adb("shell rm -f /sdcard/DCIM/Camera/xiaolv_photo.jpg")

    print("  微信: 完成!", flush=True)
    return True


def main():
    print("加载ASR...", flush=True)
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=ASR_MODEL, tokens=ASR_TOKENS, num_threads=2, language="zh", use_itn=True)
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = VAD_MODEL
    config.silero_vad.threshold = 0.2
    config.silero_vad.min_silence_duration = 0.8
    config.silero_vad.min_speech_duration = 0.2
    config.sample_rate = SAMPLE_RATE
    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=60)

    pa = pyaudio.PyAudio()
    dev_info = pa.get_device_info_by_index(MIC_DEVICE_INDEX)
    native_rate = int(dev_info['defaultSampleRate'])
    chunk_size = int(512 * native_rate / SAMPLE_RATE)
    stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=native_rate,
                     input=True, input_device_index=MIC_DEVICE_INDEX,
                     frames_per_buffer=chunk_size)

    print(f'\n{"="*40}', flush=True)
    print(f'  小绿 - 等待语音指令', flush=True)
    print(f'  说 "小绿" 唤醒', flush=True)
    print(f'  然后说 "帮我拍照发给拾捌"', flush=True)
    print(f'{"="*40}\n', flush=True)

    _cam = None
    say("小绿已就绪")

    state = "standby"  # standby → tracking → executing

    try:
        while True:
            if state == "standby":
                text = listen_once(stream, vad, recognizer, native_rate, timeout=9999)
                if text:
                    print(f"  [{text}]", flush=True)
                    if any(w in text for w in WAKE_WORDS):
                        print("\n*** 唤醒成功! ***\n", flush=True)
                        # 启动摄像头+跟踪
                        print("  启动摄像头跟踪...", flush=True)
                        ptz("track_on")
                        # 打开摄像头让追踪生效
                        _cam = cv2.VideoCapture(CAMERA_INDEX)
                        _cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        _cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        for _ in range(20): _cam.read()
                        time.sleep(1)
                        ptz("frame", 2)
                        ptz("track_on")
                        say("已启动，请说指令")
                        flush_mic(stream, vad, native_rate)
                        state = "tracking"

            elif state == "tracking":
                text = listen_once(stream, vad, recognizer, native_rate, timeout=30)
                if text is None:
                    print("  超时，关闭摄像头回到待机", flush=True)
                    if _cam: _cam.release(); _cam = None
                    ptz("center")
                    say("没听到指令，休息了")
                    flush_mic(stream, vad, native_rate)
                    state = "standby"
                    continue

                print(f"  指令: {text}", flush=True)
                import re
                if any(re.search(kw, text) for kw in PHOTO_KEYWORDS):
                    print("\n=== 执行拍照发微信 ===\n", flush=True)
                    say("好的，正在拍照")
                    flush_mic(stream, vad, native_rate)

                    # 从已打开的摄像头拍照
                    if _cam and _cam.isOpened():
                        for _ in range(10): _cam.read()
                        ret, frame = _cam.read()
                        if ret:
                            cv2.imwrite("/tmp/xiaolv_photo.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            print("  拍照OK", flush=True)
                            send_wechat("/tmp/xiaolv_photo.jpg")
                            say("照片已发给拾捌")
                        else:
                            say("拍照失败了")
                    else:
                        say("摄像头没打开")

                    # 关闭摄像头回到待机
                    if _cam: _cam.release(); _cam = None
                    ptz("center")
                    flush_mic(stream, vad, native_rate)
                    print("\n回到待机\n", flush=True)
                    state = "standby"
                else:
                    print(f"  未识别指令: {text}", flush=True)
                    say("没听懂，请说拍照发送")
                    flush_mic(stream, vad, native_rate)

    except KeyboardInterrupt:
        print("\n退出", flush=True)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


if __name__ == "__main__":
    main()
