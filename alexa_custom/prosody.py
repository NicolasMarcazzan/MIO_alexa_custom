from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime
import tokenizers as tk

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = Path(os.environ.get("PROSODY_MODEL_PATH", "models/bert-context"))

# Piper non supporta <emphasis> né <prosody rise>.
# Usiamo solo <break> e <prosody rate> che sono gli unici tag SSML
# che Piper riconosce correttamente.
#
# Enfasi: <prosody rate="slow">word</prosody> + break dopo
# Pause:  <break time="Nms"/>
# Intonazione ascendente: non supportata da Piper — saltata.

_SENTENCE_TYPE_PATTERNS: list[tuple[re.Pattern, str, float]] = [
    (re.compile(r"\?\s*$"), "question", 1.0),
    (
        re.compile(
            r"^(chi|cosa|come|quando|dove|perch[eé]|quale|quanto|che cosa)\s", re.I
        ),
        "question",
        0.9,
    ),
    (re.compile(r"^(puoi|potresti|vuoi|vuo|sa[i]|sapresti)\s", re.I), "question", 0.8),
    (re.compile(r"^\w+[!]\s*$"), "command", 0.9),
    (
        re.compile(
            r"^(dai|fai|apri|chiudi|accendi|spegni|metti|togli|vai|fermati)(\s|$)", re.I
        ),
        "command",
        0.8,
    ),
    (re.compile(r"^[.!]\s*$"), "statement", 0.7),
]

_EMPHASIS_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b[A-Z][a-z]{2,}\b"),
    re.compile(r"\b[A-Z]{2,}\b"),
    re.compile(r"\b\d+\b"),
    re.compile(r"\b(molto|davvero|assolutamente|estremamente|particolarmente)\b", re.I),
]

_PAUSE_DURATIONS = {
    ",": 150,
    ";": 200,
    ":": 200,
    ".": 400,
    "!": 400,
    "?": 400,
    "—": 250,
}

# Frasi di prototipo per classificazione BERT sentence-type via cosine-similarity.
# Vengono usate solo quando il modello ONNX è disponibile; altrimenti resta il rule-based.
_PROTOTYPE_SENTENCES: dict[str, list[str]] = {
    "question": [
        "chi sei",
        "cosa fai",
        "come stai",
        "dove vai",
        "quando arrivi",
        "perché piangi",
        "che ore sono",
        "puoi aiutarmi",
        "vuoi mangiare",
        "sai nuotare",
        "quanto costa",
        "dov'è il bagno",
    ],
    "command": [
        "apri la porta",
        "chiudi la finestra",
        "accendi la luce",
        "spegni il motore",
        "metti via",
        "togli il piatto",
        "vai via",
        "fermati adesso",
        "dai il libro",
        "fai presto",
        "chiama marco",
        "scendi le scale",
    ],
    "statement": [
        "oggi fa caldo",
        "domani piove",
        "mi piace la pizza",
        "il cielo è blu",
        "sono stanco",
        "ho fame",
        "vado a casa",
        "lui arriva tardi",
        "lei canta bene",
        "il treno parte alle otto",
    ],
}


@dataclass
class EmphasisAnnotation:
    word: str
    start: int
    end: int
    strength: float = 0.3


@dataclass
class PauseAnnotation:
    position: int
    duration_ms: int


@dataclass
class ProsodyAnnotations:
    sentence_type: str = "statement"
    confidence: float = 1.0
    emphasis: list[EmphasisAnnotation] = field(default_factory=list)
    pauses: list[PauseAnnotation] = field(default_factory=list)
    raw_text: str = ""


class ContextAnalyzer:
    _lock: threading.Lock
    _session: Any = None
    _model_path: Path
    _tokenizer: tk.Tokenizer | None = None
    _vocab: dict[str, int] | None = None
    _input_names: list[str] | None = None
    _output_names: list[str] | None = None
    _prototype_embeddings: dict[str, np.ndarray] | None = None

    def __init__(self, model_path: str | Path = _DEFAULT_MODEL_PATH) -> None:
        self._lock = threading.Lock()
        self._model_path = Path(model_path)

    def _ensure_model(self) -> bool:
        if self._session is not None:
            return True
        onnx_path = self._model_path / "model_int8.onnx"
        if not onnx_path.is_file():
            onnx_path = self._model_path / "model.onnx"
        if not onnx_path.is_file():
            return False
        try:
            self._session = onnxruntime.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )
            self._input_names = [i.name for i in self._session.get_inputs()]
            self._output_names = [o.name for o in self._session.get_outputs()]
        except Exception as e:
            logger.warning(f"Failed to load prosody model: {e}")
            return False
        self._load_tokenizer()
        self._compute_prototypes()
        return True

    def _load_tokenizer(self) -> None:
        tok_path = self._model_path / "tokenizer.json"
        if tok_path.is_file():
            try:
                self._tokenizer = tk.Tokenizer.from_file(str(tok_path))
                return
            except Exception as e:
                logger.debug(f"tokenizers load failed: {e}")
        vocab_path = self._model_path / "vocab.json"
        if vocab_path.is_file():
            try:
                with vocab_path.open() as f:
                    self._vocab = json.load(f)
            except Exception:
                pass

    def _compute_prototypes(self) -> None:
        if self._session is None:
            return
        cache_path = self._model_path / "prototypes.json"
        if cache_path.is_file():
            try:
                with cache_path.open() as f:
                    raw = json.load(f)
                self._prototype_embeddings = {
                    label: np.array(emb, dtype=np.float32) for label, emb in raw.items()
                }
                logger.debug(
                    f"Loaded {len(self._prototype_embeddings)} prototype embeddings from cache"
                )
                return
            except Exception as e:
                logger.debug(f"Failed to load prototype cache: {e}")

        embeddings: dict[str, list[np.ndarray]] = {k: [] for k in _PROTOTYPE_SENTENCES}
        for label, sentences in _PROTOTYPE_SENTENCES.items():
            for sent in sentences:
                logits = self._bert_infer(sent)
                if logits is not None:
                    cls_emb = logits[0, 0, :].copy()
                    embeddings[label].append(cls_emb)

        if any(embeddings.values()):
            self._prototype_embeddings = {}
            for label, embs in embeddings.items():
                if embs:
                    self._prototype_embeddings[label] = np.mean(embs, axis=0)
            try:
                cache = {
                    label: emb.tolist()
                    for label, emb in self._prototype_embeddings.items()
                }
                with cache_path.open("w") as f:
                    json.dump(cache, f)
                logger.debug(
                    f"Computed and cached {len(self._prototype_embeddings)} prototype embeddings"
                )
            except Exception as e:
                logger.debug(f"Failed to cache prototypes: {e}")

    def _bert_tokenize(self, text: str) -> tuple[list[int], list[int]]:
        if self._tokenizer is not None:
            try:
                encoding = self._tokenizer.encode(text)
                return encoding.ids, encoding.attention_mask
            except Exception:
                pass
        if self._vocab:
            return self._simple_wordpiece(text, self._vocab)
        return [0], [1]

    def _simple_wordpiece(
        self, text: str, vocab: dict[str, int]
    ) -> tuple[list[int], list[int]]:
        tokens: list[int] = [101]
        for char in text.lower():
            if char in vocab:
                tokens.append(vocab[char])
            else:
                tokens.append(vocab.get("[UNK]", 100))
        tokens.append(102)
        return tokens, [1] * len(tokens)

    def _classify_sentence_rule(self, text: str) -> tuple[str, float]:
        text_stripped = text.strip()
        for pattern, label, confidence in _SENTENCE_TYPE_PATTERNS:
            if pattern.search(text_stripped):
                return label, confidence
        return "statement", 0.7

    def _detect_emphasis_rule(self, text: str) -> list[EmphasisAnnotation]:
        results: list[EmphasisAnnotation] = []
        for pattern in _EMPHASIS_PATTERNS:
            for m in pattern.finditer(text):
                strength = 0.4 if m.group().isupper() else 0.3
                results.append(
                    EmphasisAnnotation(
                        word=m.group(),
                        start=m.start(),
                        end=m.end(),
                        strength=strength,
                    )
                )
        return results

    def _detect_pauses_rule(self, text: str) -> list[PauseAnnotation]:
        results: list[PauseAnnotation] = []
        for i, char in enumerate(text):
            dur = _PAUSE_DURATIONS.get(char)
            if dur is not None:
                results.append(PauseAnnotation(position=i, duration_ms=dur))
        return results

    def analyze(self, text: str) -> ProsodyAnnotations:
        if not text:
            return ProsodyAnnotations(raw_text=text)
        sentence_type, confidence = self._classify_sentence_rule(text)
        emphasis = self._detect_emphasis_rule(text)
        pauses = self._detect_pauses_rule(text)
        if self._ensure_model():
            try:
                logits = self._bert_infer(text)
                if logits is not None:
                    bert_type, bert_conf = self._bert_sentence_type(logits, text)
                    if bert_conf > confidence:
                        sentence_type = bert_type
                        confidence = bert_conf
                    bert_emphasis = self._bert_ner_emphasis(logits, text)
                    if bert_emphasis:
                        emphasis = bert_emphasis
            except Exception as e:
                logger.debug(f"BERT inference failed, using rules: {e}")
        return ProsodyAnnotations(
            sentence_type=sentence_type,
            confidence=confidence,
            emphasis=emphasis,
            pauses=pauses,
            raw_text=text,
        )

    def _bert_infer(self, text: str) -> np.ndarray | None:
        if self._session is None:
            return None
        input_ids, attention_mask = self._bert_tokenize(text)
        max_len = 128
        if len(input_ids) > max_len:
            input_ids = input_ids[: max_len - 1] + [102]
            attention_mask = attention_mask[:max_len]
        pad_len = max_len - len(input_ids)
        if pad_len > 0:
            input_ids += [0] * pad_len
            attention_mask += [0] * pad_len
        ort_inputs = {
            "input_ids": np.array([input_ids], dtype=np.int64),
            "attention_mask": np.array([attention_mask], dtype=np.int64),
            "token_type_ids": np.zeros((1, max_len), dtype=np.int64),
        }
        try:
            outputs = self._session.run(self._output_names, ort_inputs)
            return outputs[0]
        except Exception as e:
            logger.warning(f"BERT onnx inference error: {e}")
            return None

    def _bert_sentence_type(self, logits: np.ndarray, text: str) -> tuple[str, float]:
        if not self._prototype_embeddings:
            return self._classify_sentence_rule(text)
        cls_emb = logits[0, 0, :]
        cls_norm = np.linalg.norm(cls_emb)
        if cls_norm < 1e-8:
            return self._classify_sentence_rule(text)
        best_label = "statement"
        best_score = -1.0
        for label, proto in self._prototype_embeddings.items():
            proto_norm = np.linalg.norm(proto)
            if proto_norm < 1e-8:
                continue
            sim = float(np.dot(cls_emb, proto) / (cls_norm * proto_norm))
            sim = max(-1.0, min(1.0, sim))
            if sim > best_score:
                best_score = sim
                best_label = label
        confidence = max(0.5, best_score)
        return best_label, confidence

    def _bert_ner_emphasis(
        self, logits: np.ndarray, text: str
    ) -> list[EmphasisAnnotation] | None:
        if self._tokenizer is None:
            return None
        try:
            encoding = self._tokenizer.encode(text)
        except Exception:
            return None
        offsets = encoding.offsets
        if len(offsets) == 0:
            return None

        # logits shape: (1, seq_len, hidden_size)
        seq_len = logits.shape[1]
        n_tokens = min(len(offsets), seq_len - 2)
        if n_tokens == 0:
            return None

        token_norms = np.linalg.norm(logits[0, 1 : 1 + n_tokens], axis=1)
        mean_norm = float(np.mean(token_norms))
        std_norm = float(np.std(token_norms))
        if std_norm < 1e-8:
            return None
        threshold = mean_norm + 0.75 * std_norm

        results: list[EmphasisAnnotation] = []
        for i in range(n_tokens):
            if token_norms[i] > threshold:
                start, end = offsets[i]
                if end - start < 1:
                    continue
                word = text[start:end]
                strength = float(
                    min(1.0, (token_norms[i] - mean_norm) / (mean_norm + 1e-8))
                )

                # Merge with previous if it's a continuation subword (##)
                if results and results[-1].end == start and word.startswith("##"):
                    results[-1].word += word[2:]
                    results[-1].end = end
                    results[-1].strength = max(results[-1].strength, strength)
                else:
                    results.append(
                        EmphasisAnnotation(
                            word=word,
                            start=start,
                            end=end,
                            strength=strength,
                        )
                    )

        return results if results else None

    def annotate_ssml(self, text: str, lang: str = "it-IT") -> str:
        annotations = self.analyze(text)
        parts: list[str] = []
        last_end = 0
        all_markers: list[tuple[int, str, str]] = []

        for a in annotations.emphasis:
            tag = f'<prosody rate="slow">{a.word}</prosody>'
            all_markers.append((a.start, tag, "emphasis"))

        for p in annotations.pauses:
            tag = f'<break time="{p.duration_ms}ms"/>'
            all_markers.append((p.position, tag, "pause"))

        all_markers.sort(key=lambda x: x[0])

        for pos, marker, _kind in all_markers:
            if pos > last_end:
                parts.append(text[last_end:pos])
            parts.append(marker)
            last_end = pos

        if last_end < len(text):
            parts.append(text[last_end:])

        return "".join(parts)


def download_default_model(force: bool = False) -> Path:
    dest = _DEFAULT_MODEL_PATH
    if dest.is_dir() and not force:
        onnx_file = dest / "model_int8.onnx"
        if onnx_file.is_file():
            logger.info(f"Prosody model already at {dest}")
            return dest
    from huggingface_hub import hf_hub_download

    logger.info(f"Downloading BERT multilingual ONNX model to {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    try:
        hf_hub_download(
            "Xenova/bert-base-multilingual-cased",
            filename="onnx/model_int8.onnx",
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )
        src = dest / "onnx" / "model_int8.onnx"
        if src.exists():
            src.rename(dest / "model_int8.onnx")
            (dest / "onnx").rmdir()
        for fname in ("tokenizer.json", "config.json"):
            try:
                hf_hub_download(
                    "bert-base-multilingual-cased",
                    filename=fname,
                    local_dir=str(dest),
                    local_dir_use_symlinks=False,
                )
            except Exception:
                pass
        logger.info(f"Prosody model ready at {dest}")
    except Exception as e:
        logger.error(f"Failed to download prosody model: {e}")
    return dest


_analyzer: ContextAnalyzer | None = None
_analyzer_lock = threading.Lock()


def get_analyzer() -> ContextAnalyzer:
    global _analyzer
    if _analyzer is None:
        with _analyzer_lock:
            if _analyzer is None:
                _analyzer = ContextAnalyzer()
    return _analyzer


def annotate(text: str, lang: str = "it-IT") -> str:
    return get_analyzer().annotate_ssml(text, lang)
