from __future__ import annotations

import unittest

from shadowhand_gpu.rigid_flex_diagnostic import RIGID_FLEX_FIXTURES


class RigidFlexDiagnosticFixtureTests(unittest.TestCase):
    def test_required_controlled_fixture_set_is_present(self):
        self.assertEqual(
            [fixture.name for fixture in RIGID_FLEX_FIXTURES],
            [
                "free_no_contact",
                "single_geom_approach",
                "isolated_fingertip",
                "isolated_palm",
                "settled_contact",
            ],
        )
        self.assertIsNone(RIGID_FLEX_FIXTURES[0].enabled_geom)
        self.assertEqual(RIGID_FLEX_FIXTURES[1].enabled_geom, "robot0:C_ffdistal")
        self.assertEqual(RIGID_FLEX_FIXTURES[3].enabled_geom, "robot0:C_palm0")


if __name__ == "__main__":
    unittest.main()
