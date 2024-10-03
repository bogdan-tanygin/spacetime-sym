[![Build](https://github.com/bogdan-tanygin/spacetime-sym/actions/workflows/build.yml/badge.svg)](https://github.com/bogdan-tanygin/spacetime-sym/actions/workflows/build.yml)

`spacetime-sym` is a Python module introducing symmetry theory formalism into data science and natural sciences. It is a fork of [bsym][githubbsym]. It consists of core classes that describe configuration vector and Euclidean 3D spaces, their symmetry operations (e.g. `O(3)` and `SO(3)` groups), specific configurations of objects within these spaces, and physical quantities. The module also contains an interface for working with [`pymatgen`](https://pymatgen.org) `Structure` objects, to allow simple generation of disordered symmetry-inequivalent structures from a symmetric parent crystal structure.

`spacetime-sym` supports the following physical quantities: scalars, pseudoscalars, vectors, axial vectors, and tensors. They are powered by all possible dichromatic symmetry properties, such as: [parity](https://en.wikipedia.org/wiki/Parity_(physics)) `P`, [charge conjugation](https://en.wikipedia.org/wiki/C-symmetry) `C`, and [time reversal](https://en.wikipedia.org/wiki/T-symmetry) `T`. 

Source code is available as a git repository at [https://github.com/bogdan-tanygin/spacetime-sym][github].

## Installation

Clone the latest development version
```
git clone git@github.com:bogdan-tanygin/spacetime-sym.git
```
and install:
```
cd spacetime-sym
python3 setup.py install 
```

## Tests

Manual tests can be run using
```
pytest tests/unit_tests --cov-config .coveragerc --cov=spacetime --cov-report xml
```

The code has been tested with Python versions 3.9.18 and above.

## LICENSE

The `spacetime-sym` is a dual-licensed software. The sublicensing happened in 2024 as permitted by the original `bsym`'s MIT license.

### Option 1: open-source license

The `spacetime-sym` can be used as an open-source software according to the terms of GPL-3.0.

### Option 2: commercial license

After signing the written agreement with Dr. Bogdan Tanygin (info@deeptech.business), the `spacetime-sym` can be used in a proprietary software project.

## Citing

### Citing `spacetime-sym`

The publication will be added later. If it is needed now, the general methodological concept was described here:

Tanygin, Bogdan M. (2011). *Symmetry theory of the flexomagnetoelectric effect in the magnetic domain walls*. Journal of Magnetism and Magnetic Materials. https://doi.org/10.1016/j.jmmm.2010.10.028

#### BibTeX

```
@article{tanygin2011symmetry,
  title={Symmetry theory of the flexomagnetoelectric effect in the magnetic domain walls},
  author={Tanygin, Bogdan M},
  journal={Journal of Magnetism and Magnetic Materials},
  volume={323},
  number={5},
  pages={616--619},
  year={2011},
  publisher={Elsevier}
}
```

### Citing `bsym`

The original `bsym` code can be cited as:

Morgan, Benjamin J. (2017). *bsym - a Basic Symmetry Module*. The Journal of Open Source Software. http://doi.org/10.21105/joss.00370

#### BibTeX

```
@article{Morgan_JOSS2017b,
  doi = {10.21105/joss.00370},
  url = {https://doi.org/10.21105/joss.00370},
  year  = {2017},
  month = {aug},
  publisher = {The Open Journal},
  volume = {2},
  number = {16},
  author = {Benjamin J. Morgan},
  title = {bsym: A basic symmetry module},
  journal = {The Journal of Open Source Software}
}
```
[github]: https://github.com/bogdan-tanygin/spacetime-sym
