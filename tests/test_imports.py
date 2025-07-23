import importlib

for mod in ["osiris_utils.data", "osiris_utils.decks", "osiris_utils.postprocessing"]:
    assert importlib.import_module(mod)
