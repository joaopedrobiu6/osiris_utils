# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
# ---

"""
Quick demonstration of osiris_utils:
• opens an OSIRIS simulation directory
• plots Ez field at t = 2000
Run with:  python examples/quick_start.py  <PATH_TO_RUN>
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

import osiris_utils as ou

sim_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("example_data/thermal.1d")
sim = ou.Simulation(sim_path)

# grab Ez diagnostic
ez = sim["e3"]


# plot Ez field at iteration 220
plt.plot(ez.x, ez[220], label=f"${ez.label}$")
plt.title(rf"${ez.label}$ at t = {ez.time(220)[0]} $[{ez.time(220)[1]}]$")
plt.xlabel(ez.axis[0]["plot_label"])
plt.ylabel(f"${ez.units}$")
plt.legend()
plt.tight_layout()
plt.show()
