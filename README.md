[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17573059.svg)](https://doi.org/10.5281/zenodo.17573059)
[![Build](https://github.com/bogdan-tanygin/spacetime-sym/actions/workflows/build.yml/badge.svg)](https://github.com/bogdan-tanygin/spacetime-sym/actions/workflows/build.yml)
[![Coverage Status](https://coveralls.io/repos/github/bogdan-tanygin/spacetime-sym/badge.svg?branch=main)](https://coveralls.io/github/bogdan-tanygin/spacetime-sym?branch=main)

`spacetime-sym` is a Python module introducing symmetry theory formalism into data science and natural sciences. It is a fork of [bsym][githubbsym]. It consists of core classes that describe configuration vector and Euclidean 3D spaces, their symmetry operations (e.g. `O(3)` and `SO(3)` groups), specific configurations of objects within these spaces, and physical quantities.

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

Test scripts can be run using
```
pytest tests/unit_tests --cov-config .coveragerc --cov=spacetime --cov-report xml
```

The code has been tested with Python versions 3.9.18 and above.

## License

`spacetime-sym` is available under the GNU General Public License version 3 (GPLv3) or under a separate written commercial license agreement.

### Open-source license

You may use, modify, and redistribute `spacetime-sym` under the terms of the [GNU General Public License version 3](LICENSE).

### Commercial license

To license `spacetime-sym` for integration into proprietary software under commercial terms, enter into a separate written agreement with Dr. Bogdan Tanygin. See [commercial licensing information](LICENSE-commercial) or contact [info@deeptech.business](mailto:info@deeptech.business).

### Upstream project and attribution

`spacetime-sym` is based on [`bsym`](https://github.com/bjmorgan/bsym) by Benjamin J. Morgan. The original codebase has been substantially reworked and extended for space-time symmetry analysis. Some portions of the original `bsym` code remain unchanged.

The original `bsym` copyright notice and MIT license text are included in [`licenses/bsym-MIT`](licenses/bsym-MIT).

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
[githubbsym]: https://github.com/bjmorgan/bsym
