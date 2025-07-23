import ast
import copy
import re

import numpy as np

from .species import Specie


def deval(x):
    """
    Auxiliar to handle eval of Fortran formatted numbers (e.g. 1.4d-5)
    """
    if "d" in x:
        x = x.replace("d", "e")
    return float(x)


class InputDeckIO:
    """
    Class to handle parsing/re-writing of OSIRIS input decks

    Parameters
    ----------
    filename : str
        Path to OSIRIS input deck file.

    verbose : bool
        If True, prints additional information when parsing file.
        Helpful for debugging issues if input deck parsing fails.

    Attributes
    ----------
    filename : str
        Path to original input file used to create the InputDeckIO object.

    sections : list[dict]
        List of pairs (section_name: str, section_dict: dict) which contain
        current state of InputDeckIO object.

    dim : int
        Number of dimensions in the simulation (1, 2, or 3).
    """

    def __init__(self, filename: str, verbose: bool = True):
        self._filename = str(filename)
        self._sections = self._parse_input_deck(verbose)
        self._dim = self._get_dim()
        self._species = self._get_species()

    def _parse_input_deck(self, verbose):
        section_list = []

        if verbose:
            print(f"\nParsing input deck : {self._filename}")

        with open(self._filename, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # remove comments
        lines = [l[: l.find("!")] if "!" in l else l for l in lines]  # noqa: E741

        # join into single string (makes it easier to parse using regex)
        lines = "".join(lines)

        # remove tabs/spaces/paragraphs (except spaces inside "")
        lines = re.sub(r'"[^"]*"|(\s+)', lambda x: "" if x.group(1) else x.group(0), lines)

        # split sections
        # get name before brackets
        section_names = re.findall(r"(?:^|\})(.*?)(?:\{)", lines)
        # get content inside brackets
        section_infos = re.findall(r"(?:\{)(.*?)(?:\})", lines)

        if len(section_names) != len(section_infos):
            raise RuntimeError(
                "Unexpected problems parsing the document!\n"
                f"Number of section names detected ({len(section_names)}) "
                f"is different from number of sections ({len(section_infos)}).\n"
                "Might be a bug in the code, or problem with input deck format!"
                "Check if you could run the deck with OSIRIS."
            )

        # parse section information
        for section, info in zip(section_names, section_infos, strict=False):
            if verbose:
                print(f"Reading {section}")

            # split section contents at commas (unless comma inside brackets e.g. pmax(1:2, 1))
            info = re.split(r",(?![^()]*\))\s*", info)
            info = list(filter(None, info))

            # save pairs of (param, values) to dict
            section_dict = {}
            param = ""
            value = ""

            for i in info:
                aux = i.split("=")
                # solution to deal with parameters given by multiple values
                # (e.g. ps_np(1:3) = 512,128,128 which was split into:
                #   ['ps_np(1:3)=512', '128', '128'])
                # need to be able to regroup '128's with the previous value
                if len(aux) == 1 and param != "":
                    value = ",".join([value, aux[0]])
                    section_dict[param] = value
                # simplest case where we simply have "param=value"
                elif len(aux) == 2:
                    param, value = aux
                    section_dict[param] = value
                # case where we have multipel '=' inside strings
                # happens for e.g. with mathfuncs in density profiles
                else:
                    param = aux[0]
                    value = "".join(aux[1:])
                    # check that value is actually wrapped inside a string
                    if value[0] in ['"', "'"] and value[-1] in ['"', "'"]:
                        section_dict[param] = value
                    # because if not, then there is an error in the parser
                    # or a problem with the input deck
                    else:
                        raise RuntimeError(
                            f'Error parsing section: "{section}".\n'
                            "Might be a bug in the code, or problem with input deck format!\n"
                            "Check if you could run the deck with OSIRIS."
                        )

            section_list.append([section, section_dict])

            if verbose:
                for k, v in section_dict.items():
                    print(f"  {k} = {v}")

        if verbose:
            print("Input deck successfully parsed\n")

        return section_list

    def _get_dim(self):
        dim = None
        for i in range(1, 4):
            try:
                self.get_param(section="grid", param=f"nx_p(1:{i})")
            except KeyError:
                pass
            else:
                dim = i
                break
        if dim is None:
            raise RuntimeError("Error parsing grid dimension. Grid dimension could not be estabilished.")
        return dim

    def _get_species(self):
        s_names = self.get_param("species", "name")
        s_rqm = self.get_param("species", "rqm")
        # real charge is optional in OSIRIS
        # if real charge not provided assume electron charge
        try:
            s_qreal = self.get_param("species", "q_real")
            s_qreal = np.array([float(q) for q in s_qreal])
        except KeyError:
            s_qreal = np.ones(len(s_names))
        # check if we have information for all species
        if len(s_names) != self.n_species:
            raise RuntimeError("Number of specie names does not match number of species: " f"{len(s_names)} != {len(self.n_species)}.")
        if len(s_rqm) != self.n_species:
            raise RuntimeError("Number of specie rqm does not match number of species: " f"{len(s_rqm)} != {len(self.n_species)}.")
        if len(s_qreal) != self.n_species:
            raise RuntimeError("Number of specie rqm does not match number of species: " f"{len(s_qreal)} != {len(self.n_species)}.")

        return {
            ast.literal_eval(s_names[i]): Specie(
                name=ast.literal_eval(s_names[i]),
                rqm=float(s_rqm[i]),
                q=int(s_qreal[0]) * np.sign(float(s_rqm[i])),
            )
            for i in range(self.n_species)
        }

    def set_param(self, section, param, value, i_use=None, unexistent_ok=False):
        # get all sections with the same name
        # (e.g. there might be multiple 'species')
        i_sections = [i for i, m in enumerate(self._sections) if m[0] == section]

        if len(i_sections) == 0:
            raise KeyError(f'section "{section}" not found')

        if i_use is not None:
            try:
                i_sections = [im for i, im in enumerate(i_sections) if i in i_use]
            except TypeError:
                i_sections = [i_sections[i_use]]

        for i in i_sections:
            if not unexistent_ok and param not in self._sections[i][1]:
                raise KeyError(f'"{param}" not yet inside section "{section}" ' "(set unexistent_ok=True to ignore).")
            if isinstance(value, str):
                self._sections[i][1][param] = str(f'"{value}"')
            elif isinstance(value, list):
                self._sections[i][1][param] = ",".join(map(str, value))
            else:
                self._sections[i][1][param] = str(value)

    def set_tag(self, tag, value):
        for im, (_, params) in enumerate(self._sections):
            for p, v in params.items():
                self._sections[im][1][p] = v.replace(tag, str(value))

    def get_param(self, section, param):
        i_sections = [i for i, m in enumerate(self._sections) if m[0] == section]

        if len(i_sections) == 0:
            print("section not found")
            return []

        values = []
        for i in i_sections:
            if param not in self._sections[i][1]:
                raise KeyError(f'"{param}" not found inside section "{section}"')
            values.append(copy.deepcopy(self._sections[i][1][param]))

        return values

    def delete_param(self, section, param):
        sections_new = []
        for m_name, m_dict in self._sections:
            if m_name == section and param in m_dict:
                m_dict.pop(param)
            sections_new.append([m_name, m_dict])
        self._sections = sections_new

    def print_to_file(self, filename):
        with open(filename, "w", encoding="utf-8") as f:
            for section, section_dict in self._sections:
                f.write(f"{section}\n{{\n")
                for k, v in section_dict.items():
                    f.write(f'\t{k} = {v.replace(",", ", ")},\n')
                f.write("}\n\n")

    def __getitem__(self, section):
        return copy.deepcopy([m[1] for m in self._sections if m[0] == section])

    # Getters
    @property
    def filename(self):
        return self._filename

    @property
    def sections(self):
        return self._sections

    @property
    def dim(self):
        return self._dim

    @property
    def n_species(self):
        try:
            return int(self["particles"][0]["num_species"])
        except (KeyError, IndexError):
            # If num_species doesn't exist, try num_cathode
            try:
                return int(self["particles"][0]["num_cathode"])
            except (KeyError, IndexError):
                # If neither exists, raise an informative error
                raise KeyError("Could not find 'num_species' or 'num_cathode' in the particles section") from None

    @property
    def species(self):
        return self._species
