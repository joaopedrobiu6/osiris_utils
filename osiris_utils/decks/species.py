__all__ = ["Species"]


class Species:
    """
    Class to store OSIRIS species object.

    Parameters
    ----------
    name : str
        Species name.

    rqm : float
        Species charge to mass ratio.

    q : float
        Species charge in units of the elementary charge (OSIRIS's
        ``q_real``; 1 when the deck omits it).  Only its magnitude is
        used — the sign is taken from ``rqm = m/q``, so an electron is
        ``Species(rqm=-1)`` with or without ``q=-1``.

    Attributes
    ----------
    name : str
        Species name.

    rqm : float
        Species charge to mass ratio.

    q : float
        Species charge in units of the elementary charge, signed.

    m : float
        Species mass in units of the electron mass.
    """

    def __init__(self, name, rqm, q: float = 1.0):
        if q == 0:
            raise ValueError(f"Species '{name}' has q = 0: the number density (charge / q) is undefined.")
        self._name = name
        self._rqm = float(rqm)
        # Only the MAGNITUDE of q is taken; the sign is the sign of rqm = m/q,
        # since m > 0.  Deriving it here instead of trusting a signed argument
        # is what keeps the two from disagreeing: Species(rqm=-1) is an electron
        # whether or not the caller also passed q=-1, and the density (charge/q)
        # comes out negative either way.
        self._q = abs(float(q)) * (1.0 if self._rqm >= 0 else -1.0)
        self._m = self._rqm * self._q  # = |rqm| * |q| > 0

    def __repr__(self) -> str:
        return f"Species(name={self._name}, rqm={self._rqm}, q={self._q}, m={self._m})"

    @property
    def name(self):
        return self._name

    @property
    def rqm(self):
        return self._rqm

    @property
    def q(self):
        return self._q

    @property
    def m(self):
        return self._m
