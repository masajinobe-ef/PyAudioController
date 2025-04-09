import json
import time
import pyaudio
import aubio
import numpy as np
from pynput import mouse
from pynput.keyboard import Controller as KeyController
from pynput.mouse import Controller as MouseController

class OptimizedAudioController:
    def __init__(self, config_path='config.json'):
        self.load_config(config_path)
        self.setup_audio()
        self.setup_controllers()
        self.last_processed_time = 0
        self.activation_history = {}

    def load_config(self, path):
        with open(path, 'r') as f:
            self.config = json.load(f)

        self.note_map = {}
        for note, params in self.config['note_bindings'].items():
            self.note_map[params['freq']] = {
                "type": params['type'],
                "action": params['action']
            }

    def setup_audio(self):
        self.p = pyaudio.PyAudio()
        self.set_pitch_algorithm()

        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.config['audio_settings']['sample_rate'],
            input=True,
            input_device_index=self.get_input_device_index(),
            frames_per_buffer=self.config['audio_settings']['chunk_size'],
            start=False,
            stream_callback=self.audio_callback
        )

    def set_pitch_algorithm(self):
        algorithm = self.config['audio_settings'].get('pitch_algorithm', 'yin')
        self.pitch_detector = aubio.pitch(
            algorithm,
            self.config['audio_settings']['chunk_size'],
            self.config['audio_settings']['chunk_size'],
            self.config['audio_settings']['sample_rate']
        )
        self.pitch_detector.set_silence(self.config['audio_settings']['silence_threshold'])
        self.pitch_detector.set_tolerance(0.7)

        if algorithm == 'yinfft':
            self.pitch_detector.set_tolerance(0.85)

    def get_input_device_index(self):
        try:
            default_dev = self.p.get_default_input_device_info()
            return default_dev['index']
        except Exception:
            for i in range(self.p.get_device_count()):
                dev = self.p.get_device_info_by_index(i)
                if dev['maxInputChannels'] > 0:
                    return i
        return None

    def setup_controllers(self):
        self.keyboard = KeyController()
        self.mouse = MouseController()
        self.key_states = {}
        self.mouse_states = {'left': False, 'right': False}
        self.cpu_profile = self.config['audio_settings'].get('cpu_usage', 'medium')

    def optimize_cpu_usage(self):
        if self.cpu_profile == 'low':
            aubio.filter.set_resampler_quality(self.config['audio_settings'].get('resample_quality', 3))
        elif self.cpu_profile == 'high':
            aubio.filter.set_resampler_quality(self.config['audio_settings'].get('resample_quality', 5))

    def find_closest_note(self, freq):
        try:
            return min(self.note_map.keys(), key=lambda x: abs(float(x) - freq))
        except Exception:
            return None

    def handle_action(self, freq, confidence):
        if confidence < self.config['audio_settings']['confidence_threshold']:
            self.release_all()
            return

        closest_freq = self.find_closest_note(freq)
        if closest_freq is None:
            return

        note_config = self.note_map[closest_freq]

        current_time = time.time()
        min_interval = 0.1 if self.cpu_profile == 'low' else 0.05

        if current_time - self.last_processed_time < min_interval:
            return

        self.last_processed_time = current_time

        if note_config['type'] == 'key':
            self.handle_key(note_config['action'])
        elif note_config['type'] == 'mouse':
            self.handle_mouse(note_config['action'])

    def handle_key(self, key):
        if key not in self.key_states or not self.key_states.get(key, False):
            self.keyboard.press(key)
            self.key_states[key] = True

    def handle_mouse(self, button):
        if not self.mouse_states.get(button, False):
            getattr(self.mouse, 'press')(getattr(mouse.Button, button))
            self.mouse_states[button] = True

    def release_all(self):
        for key in list(self.key_states.keys()):
            if self.key_states[key]:
                self.keyboard.release(key)
                self.key_states[key] = False

        for button in ['left', 'right']:
            if self.mouse_states.get(button, False):
                getattr(self.mouse, 'release')(getattr(mouse.Button, button))
                self.mouse_states[button] = False

    def audio_callback(self, in_data, frame_count, time_info, status):
        samples = np.frombuffer(in_data, dtype=np.float32)
        pitch = self.pitch_detector(samples)[0]
        conf = self.pitch_detector.get_confidence()

        if self.cpu_profile == 'low':
            pitch = round(pitch, 1)

        self.handle_action(pitch, conf)
        return (in_data, pyaudio.paContinue)

    def run(self):
        print("🎸 Controller started!")
        self.optimize_cpu_usage()

        try:
            self.stream.start_stream()
            while self.stream.is_active():
                if self.cpu_profile == 'low':
                    time.sleep(0.05)
                elif self.cpu_profile == 'medium':
                    time.sleep(0.03)
                else:
                    time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n🛑 Shutdown...")
        finally:
            self.release_all()
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()

if __name__ == "__main__":
    controller = OptimizedAudioController()
    controller.run()

