import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from risk_model.records import Registry, ValidationError  # noqa: E402


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "t.db"
        self.reg = Registry(self.path)

    def tearDown(self):
        self.reg.close()
        self.tmp.cleanup()

    def test_add_and_get(self):
        rec = self.reg.add("Ana", 17, "Science", "ana@example.com", ["Math"])
        self.assertEqual(rec.record_id, "S0001")
        self.assertEqual(self.reg.get("S0001").courses, ["Math"])

    def test_ids_survive_reopen(self):
        self.reg.add("A", 15, "", "a@x.io")
        self.reg.close()
        self.reg = Registry(self.path)
        self.assertEqual(self.reg.add("B", 16, "", "b@x.io").record_id, "S0002")

    def test_bad_age_rejected(self):
        for age in (0, -3, "abc", 400):
            with self.assertRaises(ValidationError):
                self.reg.add("X", age, "", "x@x.io")

    def test_bad_and_duplicate_email(self):
        with self.assertRaises(ValidationError):
            self.reg.add("X", 20, "", "nope")
        self.reg.add("X", 20, "", "x@x.io")
        with self.assertRaises(ValidationError):
            self.reg.add("Y", 21, "", "x@x.io")

    def test_courses_and_delete(self):
        self.reg.add("X", 20, "", "x@x.io", ["A"])
        self.reg.set_courses("S0001", add=["B"], remove=["A"])
        self.assertEqual(self.reg.get("S0001").courses, ["B"])
        self.assertTrue(self.reg.delete("S0001"))
        self.assertIsNone(self.reg.get("S0001"))


if __name__ == "__main__":
    unittest.main()
