import json
import time
import numpy as np
import sounddevice as sd
from pynput.keyboard import Controller as KeyController
from pynput.mouse import Controller as MouseController, Button
from collections import deque

class GuitarHeroController:
    def __init__(self, config_path="config.json"):
        self.load_config(config_path)
        self.device_index = self.select_input_device()
        self.setup_audio_processing()
        self.setup_controllers()
        self.window = np.blackman(self.chunk_size)
        self.freq_history = deque(maxlen=10)
        self.note_display = {
            82.41: ("E2", "W [↑]", "↑"),
            110.0: ("A2", "S [↓]", "↓"),
            146.83: ("D3", "A [←]", "←"),
            196.0: ("G3", "D [→]", "→"),
            246.94: ("B3", "LMB [⚔]", "⚔"),
            329.63: ("E4", "RMB [🛡]", "🛡")
        }
        self.active_actions = set()
        self.spectrum = []
        self.last_print = time.time()

    def load_config(self, path):
        with open(path, "r") as f:
            self.config = json.load(f)
        self.note_map = {params["freq"]: params for params in self.config["note_bindings"].values()}
        self.detection_thresh = self.config["audio_settings"]["detection_threshold"]
        print("🎸 Гитарный контроллер инициализирован!")

    def select_input_device(self):
        devices = sd.query_devices()
        input_devices = [i for i, dev in enumerate(devices) if dev["max_input_channels"] > 0]
        if not input_devices:
            raise RuntimeError("🔇 Устройства ввода не найдены")
        print("\n🎤 Доступные аудиоустройства:")
        for i in input_devices:
            print(f"  [{i}] {devices[i]['name']}")
        while True:
            try:
                choice = int(input("🎚 Выберите номер устройства: "))
                if choice in input_devices:
                    return choice
                print("❌ Неверный выбор")
            except ValueError:
                print("⚠️ Введите число")

    def setup_audio_processing(self):
        audio_cfg = self.config["audio_settings"]
        device_info = sd.query_devices(self.device_index)
        self.sample_rate = int(audio_cfg.get("sample_rate", device_info["default_samplerate"]))
        self.chunk_size = audio_cfg["chunk_size"]
        self.silence_thresh = audio_cfg["silence_threshold"]
        self.stream = sd.InputStream(
            device=self.device_index,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            callback=self.process_audio,
            dtype="float32"
        )
        print(f"🔊 Аудиопоток настроен на {self.sample_rate/1000}kHz")

    def setup_controllers(self):
        self.keyboard = KeyController()
        self.mouse = MouseController()
        print("🎮 Контроллеры подключены")

    def process_audio(self, indata, frames, time_info, status):
        audio_data = indata[:, 0] * np.iinfo(np.int16).max
        audio_data = audio_data.astype(np.float32)
        windowed = audio_data * self.window
        fft = np.fft.rfft(windowed)
        magnitudes = np.abs(fft) ** 2
        min_freq = 50
        min_bin = int(min_freq * self.chunk_size / self.sample_rate)
        min_bin = max(1, min_bin)
        spectrum_segment = magnitudes[min_bin:int(len(magnitudes)*0.8)]
        peak_bin = np.argmax(spectrum_segment) + min_bin
        peak_power = magnitudes[peak_bin]
        if peak_power < self.detection_thresh:
            self.release_actions()
            return
        freq = peak_bin * self.sample_rate / self.chunk_size
        self.freq_history.append(freq)
        if 1 < peak_bin < len(magnitudes)-1:
            y0, y1, y2 = np.log(magnitudes[peak_bin-1:peak_bin+2])
            delta = (y0 - y2) / (2*(y0 - 2*y1 + y2))
            refined_freq = (peak_bin + delta) * self.sample_rate / self.chunk_size
            freq = refined_freq
        detected_note = self.find_closest_note(freq)
        self.spectrum = magnitudes
        if time.time() - self.last_print > 0.1:
            self.visualize_spectrum()
            self.last_print = time.time()
        if detected_note:
            self.trigger_action(detected_note)
            self.display_note(detected_note)
        else:
            self.release_actions()

    def find_closest_note(self, freq):
        frequencies = np.array(list(self.note_map.keys()))
        if frequencies.size == 0 or freq < 50 or freq > 1000:
            return None
        ratios = frequencies / freq
        cents = 1200 * np.log2(ratios)
        closest_idx = np.argmin(np.abs(cents))
        for i, note_freq in enumerate(frequencies):
            for harmonic in [2, 3, 0.5]:
                if abs(freq - note_freq*harmonic) < 10:
                    return note_freq
        return frequencies[closest_idx] if abs(cents[closest_idx]) < 50 else None

    def visualize_spectrum(self):
        BANDS = 30
        max_freq = 1000
        max_bin = int(max_freq * self.chunk_size / self.sample_rate)
        spectrum = np.log(self.spectrum[:max_bin] + 1e-10)
        chunk_size = len(spectrum) // BANDS
        if chunk_size == 0:
            return
        bars = []
        for i in range(BANDS):
            band = spectrum[i*chunk_size:(i+1)*chunk_size]
            bars.append(int(np.mean(band) * 2))
        scale = "▁▂▃▄▅▆▇"
        visualization = "".join([scale[min(v//2, len(scale)-1)] for v in bars])
        print(f"\033[F\033[K{visualization}")

    def trigger_action(self, freq):
        action = self.note_map[freq]
        action_id = f"{action['type']}_{action['action']}"
        if action_id not in self.active_actions:
            if action['type'] == 'key':
                self.keyboard.press(action['action'])
            elif action['type'] == 'mouse':
                btn = Button.left if action['action'] == 'left' else Button.right
                self.mouse.press(btn)
            self.active_actions.add(action_id)

    def release_actions(self):
        for action_id in list(self.active_actions):
            parts = action_id.split('_')
            if parts[0] == 'key':
                self.keyboard.release(parts[1])
            elif parts[0] == 'mouse':
                btn = Button.left if parts[1] == 'left' else Button.right
                self.mouse.release(btn)
            self.active_actions.remove(action_id)

    def display_note(self, freq):
        note_info = self.note_display.get(freq, ("", "", ""))
        visual = f"""
        ╔══{'═'*len(note_info[2])}══╗
        ║  {note_info[2]}  ║
        ╚══{'═'*len(note_info[2])}══╝
        {note_info[1]}
        """
        print(f"\033[F\033[K🎶 Нота: {note_info[0]} | Действие: {visual}")

    def run(self):
        print("\n🎮 Готов к игре! Нажмите Ctrl+C для выхода")
        try:
            with self.stream:
                print("\n" * 5)
                while True:
                    time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n🛑 Выключаю контроллер...")
            self.release_actions()

if __name__ == "__main__":
    controller = GuitarHeroController()
    controller.run()
