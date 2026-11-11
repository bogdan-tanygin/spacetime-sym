# spacetime-sym

[![Build](https://github.com/bogdan-tanygin/spacetime-sym/actions/workflows/build.yml/badge.svg)](https://github.com/bogdan-tanygin/spacetime-sym/actions/workflows/build.yml)
[![Documentation](https://github.com/bogdan-tanygin/spacetime-sym/actions/workflows/docs.yml/badge.svg?branch=main)](https://bogdan-tanygin.github.io/spacetime-sym/)
[![Coverage](https://coveralls.io/repos/github/bogdan-tanygin/spacetime-sym/badge.svg?branch=main)](https://coveralls.io/github/bogdan-tanygin/spacetime-sym?branch=main)

![Abstract visualization of spatial symmetry using translucent surfaces](https://raw.githubusercontent.com/bogdan-tanygin/spacetime-sym/main/docs/source/_static/images/symmetry-hero.png)

**Spatial and dichromatic point groups, Curie limiting groups, and invariance of physical quantities in Python.**

`spacetime-sym` is a scientific Python library for constructing **spatial and dichromatic point groups** and testing the **invariance of physical quantities** under their operations. It combines rotations and reflections in **O(3)** (including proper rotations in **SO(3)**) with discrete reversal operations such as **time reversal (T)**, and supports selected continuous **Curie limiting groups**. Physical quantities carry explicit transformation characters, including the spatial **P** convention and a symbolic **C** label. The supported representations include **scalars, pseudoscalars, polar and axial vectors, and second-rank tensors**. The emphasis is on symmetry constraints **beyond periodic crystal structures**.

### Highlights

- **Spatial and dichromatic point groups:** construct finite groups from rotations, reflections, and generators combined with time reversal.
- **Physical-quantity invariance:** test scalars, pseudoscalars, polar and axial vectors, and second-rank tensors with their specified spatial and dichromatic transformation characters.
- **Continuous Curie limiting groups:** check supported invariants under selected infinite axial symmetries without sampling every rotation angle.
- **Reproducible examples:** finite and limiting-group cases, including tensor invariance, in the [documentation](https://bogdan-tanygin.github.io/spacetime-sym/) and [unit tests](https://github.com/bogdan-tanygin/spacetime-sym/tree/main/tests/unit_tests/).

The current `C` label is a simplified dichromatic operation; **Dirac-spinor charge conjugation is not implemented**.

## Installation

**Python 3.12+** is required.

```bash
git clone https://github.com/bogdan-tanygin/spacetime-sym.git
cd spacetime-sym
python -m pip install -e .
```

## Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/unit_tests
```

## Documentation and research

- [Scientific overview and API documentation](https://bogdan-tanygin.github.io/spacetime-sym/)
- [Executable examples in the test suite](https://github.com/bogdan-tanygin/spacetime-sym/tree/main/tests/unit_tests/)
- [Research background: Tanygin's thesis](https://doi.org/10.5281/zenodo.17573059) and [Hlinka's vectorlike-quantity classification](https://doi.org/10.1103/PhysRevLett.113.165502)

## License, attribution, and contributions

<a href="http://deeptech.business" title="Deeptech">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/bogdan-tanygin/spacetime-sym/main/docs/source/_static/images/dt_logo_white.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/bogdan-tanygin/spacetime-sym/main/docs/source/_static/images/dt_logo_black.png">
    <img src="https://raw.githubusercontent.com/bogdan-tanygin/spacetime-sym/main/docs/source/_static/images/dt_logo_black.png" alt="Deeptech logo" width="120">
  </picture>
</a>

`spacetime-sym` is available under [GPL-3.0-only](https://github.com/bogdan-tanygin/spacetime-sym/blob/main/LICENSE) **or** a [separate written commercial license](https://github.com/bogdan-tanygin/spacetime-sym/blob/main/LICENSE-commercial). For commercial use or research partnerships, contact [info@deeptech.business](mailto:info@deeptech.business).

The project builds on [bsym](https://github.com/bjmorgan/bsym) by Benjamin J. Morgan ([JOSS, 2017](https://doi.org/10.21105/joss.00370)); retained upstream code is covered by the original [MIT notices](https://github.com/bogdan-tanygin/spacetime-sym/blob/main/licenses/bsym-MIT).

Issues, tests, documentation improvements, and pull requests are welcome. Please coordinate licensing of substantial code contributions in advance. See [Issues](https://github.com/bogdan-tanygin/spacetime-sym/issues) and [Pull requests](https://github.com/bogdan-tanygin/spacetime-sym/pulls).
