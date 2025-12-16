import unittest
from unittest.mock import MagicMock, patch

from osiris_utils.data.diagnostic import Diagnostic
from osiris_utils.data.simulation import Simulation, Species_Handler


class TestSimulation(unittest.TestCase):
    def setUp(self):
        # Mock InputDeckIO
        self.mock_deck_patcher = patch("osiris_utils.data.simulation.InputDeckIO")
        self.mock_deck_cls = self.mock_deck_patcher.start()
        self.mock_deck = self.mock_deck_cls.return_value

        # Setup mock species in input deck
        self.mock_deck.species = {"electrons": MagicMock(), "ions": MagicMock()}

        self.sim_path = "/path/to/sim/osiris.inp"
        self.sim = Simulation(self.sim_path)

    def tearDown(self):
        self.mock_deck_patcher.stop()

    def test_init(self):
        """Test simulation initialization."""
        self.assertEqual(self.sim._simulation_folder, "/path/to/sim")
        self.assertEqual(self.sim.species, ["electrons", "ions"])
        self.mock_deck_cls.assert_called_with(self.sim_path, verbose=False)

    def test_getitem_species(self):
        """Test accessing species handler."""
        handler = self.sim["electrons"]
        self.assertIsInstance(handler, Species_Handler)
        self.assertEqual(handler.species, self.mock_deck.species["electrons"])

        # Test caching
        handler2 = self.sim["electrons"]
        self.assertIs(handler, handler2)

    @patch("osiris_utils.data.simulation.Diagnostic")
    def test_getitem_diagnostic(self, mock_diag_cls):
        """Test accessing a non-species diagnostic."""
        mock_diag = mock_diag_cls.return_value

        diag = self.sim["b3"]

        mock_diag_cls.assert_called_with(simulation_folder="/path/to/sim", species=None, input_deck=self.mock_deck)
        mock_diag.get_quantity.assert_called_with("b3")

        # Verify it's cached
        diag2 = self.sim["b3"]
        self.assertIs(diag, diag2)

    def test_add_diagnostic(self):
        """Test adding custom diagnostic."""
        mock_diag = MagicMock(spec=Diagnostic)
        self.sim.add_diagnostic(mock_diag, "custom")

        self.assertIn("custom", self.sim.loaded_diagnostics)
        self.assertIs(self.sim["custom"], mock_diag)

    def test_delete_diagnostic(self):
        """Test deleting diagnostic."""
        mock_diag = MagicMock(spec=Diagnostic)
        self.sim.add_diagnostic(mock_diag, "custom")

        self.sim.delete_diagnostic("custom")
        self.assertNotIn("custom", self.sim.loaded_diagnostics)


class TestSpeciesHandler(unittest.TestCase):
    def setUp(self):
        self.handler = Species_Handler("/path/to/sim", "electrons", MagicMock())

    @patch("osiris_utils.data.simulation.Diagnostic")
    def test_getitem(self, mock_diag_cls):
        """Test getting species diagnostic."""
        mock_diag = mock_diag_cls.return_value

        _ = self.handler["charge"]

        mock_diag_cls.assert_called_with(simulation_folder="/path/to/sim", species="electrons", input_deck=self.handler._input_deck)
        mock_diag.get_quantity.assert_called_with("charge")


if __name__ == "__main__":
    unittest.main()
