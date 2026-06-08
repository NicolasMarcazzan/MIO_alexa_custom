"""Tests for prosody.py — SSML annotation, sentence-type classification, emphasis, pauses.

All tests run via the rule-based path (no BERT model needed).
"""

import unittest

from alexa_custom.prosody import (
    ContextAnalyzer,
    _PAUSE_DURATIONS,
)

_analyzer = ContextAnalyzer()


class TestSentenceTypeRules(unittest.TestCase):
    def test_question_ending_with_question_mark(self):
        # Matches pattern: r"\?\s*$"
        for text in ("come stai?", "dove vai?", "che ore sono?"):
            with self.subTest(text=text):
                st, conf = _analyzer._classify_sentence_rule(text)
                self.assertEqual(st, "question")
                self.assertEqual(conf, 1.0)

    def test_question_starting_with_interrogative(self):
        for text in (
            "chi sei",
            "cosa fai",
            "come stai",
            "dove vai",
            "quando arrivi",
            "perché piangi",
            "che cosa vuoi",
            "quale libro",
            "quanto costa",
        ):
            with self.subTest(text=text):
                st, conf = _analyzer._classify_sentence_rule(text)
                self.assertEqual(st, "question")
                self.assertGreaterEqual(conf, 0.8)

    def test_question_with_polite_verb(self):
        for text in (
            "puoi aiutarmi",
            "potresti aprire",
            "vuoi mangiare",
            "sai nuotare",
            "sapresti dirmi",
        ):
            with self.subTest(text=text):
                st, conf = _analyzer._classify_sentence_rule(text)
                self.assertEqual(st, "question")
                self.assertGreaterEqual(conf, 0.8)

    def test_command_ending_with_exclamation(self):
        st, conf = _analyzer._classify_sentence_rule("vai!")
        self.assertEqual(st, "command")
        self.assertEqual(conf, 0.9)

    def test_command_starting_with_imperative(self):
        for text in (
            "dai il libro",
            "fai presto",
            "apri la porta",
            "chiudi la finestra",
            "accendi la luce",
            "spegni il motore",
            "metti via",
            "togli il piatto",
            "vai via",
            "fermati",
        ):
            with self.subTest(text=text):
                st, conf = _analyzer._classify_sentence_rule(text)
                self.assertEqual(st, "command")
                self.assertGreaterEqual(conf, 0.8)

    def test_command_standalone_imperative_no_space(self):
        st, conf = _analyzer._classify_sentence_rule("fermati")
        self.assertEqual(st, "command")

    def test_statement_default(self):
        for text in (
            "oggi fa caldo",
            "domani piove",
            "il cielo è blu",
            "mi chiamo marco",
            "vado a casa",
        ):
            with self.subTest(text=text):
                st, conf = _analyzer._classify_sentence_rule(text)
                self.assertEqual(st, "statement")
                self.assertEqual(conf, 0.7)

    def test_empty_text_falls_to_statement(self):
        st, conf = _analyzer._classify_sentence_rule("")
        self.assertEqual(st, "statement")
        self.assertEqual(conf, 0.7)


class TestEmphasisRules(unittest.TestCase):
    def test_titlecase_word_detected(self):
        results = _analyzer._detect_emphasis_rule("Ciao MONDO bello")
        words = [a.word for a in results]
        self.assertIn("Ciao", words)
        self.assertIn("MONDO", words)

    def test_allcaps_word_detected(self):
        results = _analyzer._detect_emphasis_rule("ciao MONDO")
        self.assertEqual(len(results), 1)
        a = results[0]
        self.assertEqual(a.word, "MONDO")
        self.assertEqual(a.start, 5)
        self.assertEqual(a.end, 10)

    def test_number_detected(self):
        results = _analyzer._detect_emphasis_rule("alle 15 e 30")
        words = [a.word for a in results]
        self.assertIn("15", words)
        self.assertIn("30", words)

    def test_intensifier_detected(self):
        for word in (
            "molto",
            "davvero",
            "assolutamente",
            "estremamente",
            "particolarmente",
        ):
            results = _analyzer._detect_emphasis_rule(f"è {word} bello")
            words = [a.word for a in results]
            self.assertIn(word, words)

    def test_no_false_positives_on_short_words(self):
        results = _analyzer._detect_emphasis_rule("ciao come va")
        self.assertEqual(len(results), 0)


class TestPauseRules(unittest.TestCase):
    def test_comma_pause(self):
        results = _analyzer._detect_pauses_rule("ciao, mondo")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].position, 4)
        self.assertEqual(results[0].duration_ms, _PAUSE_DURATIONS[","])

    def test_period_pause(self):
        results = _analyzer._detect_pauses_rule("ciao. mondo")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].duration_ms, _PAUSE_DURATIONS["."])

    def test_multiple_pauses(self):
        results = _analyzer._detect_pauses_rule("ciao, mondo. come stai?")
        self.assertEqual(len(results), 3)

    def test_no_punctuation_no_pauses(self):
        results = _analyzer._detect_pauses_rule("ciao mondo come stai")
        self.assertEqual(len(results), 0)


class TestAnnotationSSML(unittest.TestCase):
    def test_no_forbidden_tags(self):
        result = _analyzer.annotate_ssml("ciao, mondo. come stai?")
        self.assertNotIn("<emphasis", result)
        self.assertNotIn("</emphasis>", result)
        self.assertNotIn("prosody rise", result)

    def test_break_tags_present_for_punctuation(self):
        result = _analyzer.annotate_ssml("ciao, mondo.")
        self.assertIn('<break time="', result)

    def test_emphasis_uses_prosody_rate(self):
        result = _analyzer.annotate_ssml("è MOLTO bello")
        self.assertIn('<prosody rate="slow">MOLTO</prosody>', result)

    def test_prosody_rate_for_allcaps(self):
        result = _analyzer.annotate_ssml("Ciao GIOVANNI")
        self.assertIn('<prosody rate="slow">Ciao</prosody>', result)
        self.assertIn('<prosody rate="slow">GIOVANNI</prosody>', result)

    def test_annotate_preserves_original_text_around_markers(self):
        result = _analyzer.annotate_ssml("ciao.")
        self.assertIn("ciao", result)

    def test_empty_text_returns_empty(self):
        result = _analyzer.annotate_ssml("")
        self.assertEqual(result, "")

    def test_no_tags_when_no_punctuation_or_emphasis(self):
        result = _analyzer.annotate_ssml("ciao mondo")
        self.assertEqual(result, "ciao mondo")


class TestAnalyze(unittest.TestCase):
    def test_analyze_returns_annotations(self):
        ann = _analyzer.analyze("ciao, mondo!")
        self.assertEqual(ann.raw_text, "ciao, mondo!")
        self.assertIn(ann.sentence_type, ("command", "statement"))
        self.assertGreater(len(ann.pauses), 0)

    def test_analyze_empty(self):
        ann = _analyzer.analyze("")
        self.assertEqual(ann.sentence_type, "statement")
        self.assertEqual(len(ann.emphasis), 0)
        self.assertEqual(len(ann.pauses), 0)


if __name__ == "__main__":
    unittest.main()
