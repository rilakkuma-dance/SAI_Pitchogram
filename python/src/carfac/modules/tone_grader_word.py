from modules.tone_detection_word import ToneClassifierTester as tdw
from dataclasses import dataclass
import csv

@dataclass
class GradingResult:
    chinese_text: str
    pinyin_text: str
    english_text: str
    reference_tone: str
    detected_tone: str
    confidence: float
    is_correct: bool

class ToneGraderWord:
    def __init__(self):
        self.tone_classifier = tdw()
        self.results: list[GradingResult] = list()

    # only works for monosyllabic words...
    def grade_audio(self, audio_data, reference_text: str, reference_pinyin: str, reference_english: str, reference_tone: int):
        # Compare detected tones with reference tones
        grade_result = self.tone_classifier.predict_tone(audio_data, sr=16000)
        detected_tone = grade_result['predicted_tone']
        confidence = grade_result['confidence']

        result = GradingResult(
            chinese_text=reference_text,
            pinyin_text=reference_pinyin,
            english_text=reference_english,
            reference_tone=reference_tone,
            detected_tone=detected_tone,
            confidence=confidence,
            is_correct=(detected_tone == reference_tone)
        )
        self.results.append(result)

        return result

    def save_results(self, filename: str):
        print("Saving grading results to", filename)
        writer = csv.writer(open(filename, 'w'))
        writer.writerow(['Chinese', 'Pinyin', 'English', 'Reference Tone', 'Detected Tone', 'Confidence', 'Is Correct'])
        for res in self.results:
            writer.writerow([res.chinese_text, res.pinyin_text, res.english_text, res.reference_tone, res.detected_tone, f"{res.confidence:.2f}", res.is_correct])