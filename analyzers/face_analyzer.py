import threading
from deepface import DeepFace

class FaceAnalyzer:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest = {}
        self._running = False

    def analyze_async(self, frame):
        if self._running:
            return
        self._running = True
        t = threading.Thread(target=self._run, args=(frame.copy(),), daemon=True)
        t.start()

    def _run(self, frame):
        try:
            results = DeepFace.analyze(
                frame,
                actions=['age', 'gender', 'emotion', 'race'],
                enforce_detection=False,
                silent=True
            )
            face = results[0]
            with self._lock:
                self._latest = {
                    'age':     face['age'],
                    'gender':  face['dominant_gender'],
                    'emotion': face['dominant_emotion'],
                    'race':    face['dominant_race'],
                    'region':  face['region'],
                }
        except Exception:
            pass
        finally:
            self._running = False

    def get_latest(self):
        with self._lock:
            return self._latest.copy()