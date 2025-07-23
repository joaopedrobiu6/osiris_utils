class Specie:
    """
    Class to store OSIRIS species object.

    Parameters
    ----------
    name : str
        Specie name.

    rqm : float
        Specie charge to mass ratio.

    q : int
        Specie charge in units of the electron charge.
        Electrons would be represented by q=-1 and protons q=1.

    Attributes
    ----------
    name : str
        Specie name.

    rqm : float
        Specie charge to mass ratio.

    q : int
        Specie charge in units of the electron charge.

    m : float
        Specie mass in units of the electron mass.
    """

    def __init__(self, name, rqm, q: int = 1):
        self._name = name
        self._rqm = rqm
        self._q = q
        self._m = rqm * q

    def __repr__(self) -> str:
        return f"Specie(name={self._name}, rqm={self._rqm}, q={self._q}, m={self._m})"

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
