import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import tempfile

from lexicon_g2p import LETTER_NAMES, Pronouncer, main, normalize, split_identifier, variants


class LexiconG2pTests(unittest.TestCase):
    def setUp(self):
        self.pronouncer = Pronouncer({
            "cube": [["K", "Y", "UW1", "B"]],
            "control": [["K", "AH0", "N", "T", "R", "OW1", "L"]],
            "kube": [["K", "UW", "B"]],
            "use": [["Y", "UW", "Z"]],
            "effect": [["IH", "F", "EH", "K", "T"]],
        }, "none")

    def test_user_forms_are_not_capped(self):
        result = variants("kubectl", ["cube control", "kube control"], self.pronouncer, 1)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(source == "user" for source, _ in result))
        self.assertEqual(result[0][1], ("K", "Y", "UW", "B", "K", "AH", "N", "T", "R", "OW", "L"))

    def test_identifier_splitting(self):
        self.assertEqual(split_identifier("useEffect"), ["use", "Effect"])
        self.assertEqual(split_identifier("CI/CD2"), ["CI", "slash", "CD", "two"])
        self.assertEqual(split_identifier(".env", False), ["env"])
        self.assertEqual(variants("useEffect", [], self.pronouncer)[0][1], ("Y", "UW", "Z", "IH", "F", "EH", "K", "T"))

    def test_stress_and_invalid_phones(self):
        self.assertEqual(LETTER_NAMES["x"], "ex")
        self.assertEqual(LETTER_NAMES["z"], "zee")
        self.assertEqual(normalize(["AH0", "EY1"]), ("AH", "EY"))
        with self.assertRaises(ValueError):
            normalize(["not-a-phone"])

    def test_unknown_acronym_can_use_letter_names(self):
        pronouncer = Pronouncer({"ess": [["EH", "S"]], "cue": [["K", "Y", "UW"]], "ell": [["EH", "L"]]}, "none")
        self.assertEqual(variants("SQL", [], pronouncer), [("acronym", ("EH", "S", "K", "Y", "UW", "EH", "L"))])

    def test_compound_acronym_is_one_complete_variant(self):
        pronouncer = Pronouncer({"see": [["S", "IY"]], "eye": [["AY"]], "dee": [["D", "IY"]]}, "none")
        self.assertEqual(variants("CI/CD", [], pronouncer), [("acronym", ("S", "IY", "AY", "S", "IY", "D", "IY"))])

    def test_command_line_exports_atomically_with_one_or_two_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "terms.tsv"
            source.write_text("kubectl\tcube control\t1.0\n!ignored\n", encoding="utf-8")
            for workers in (1, 2):
                output = directory / f"variants-{workers}.tsv"
                query = directory / f"query-{workers}.tsv"
                arguments = ["lexicon_g2p", str(source), str(output), "--backend", "none", "--workers", str(workers), "--export-query-dict", str(query)]
                with patch("sys.argv", arguments), patch.dict("sys.modules", {"cmudict": SimpleNamespace(dict=lambda: self.pronouncer.dictionary)}):
                    main()
                self.assertIn("kubectl\tuser\tK Y UW B K AH N T R OW L", output.read_text())
                self.assertIn("cube\tK Y UW B", query.read_text())
            self.assertEqual((directory / "variants-1.tsv").read_bytes(), (directory / "variants-2.tsv").read_bytes())


if __name__ == "__main__":
    unittest.main()