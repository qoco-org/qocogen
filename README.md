# QOCOGEN
<p align="center">
  <img src="https://github.com/user-attachments/assets/7bd44fa7-d198-4739-bb79-a5c15e04a8de" alt="drawing" width="500"/>
</p>

<p align="center">
  <a href=https://github.com/qoco-org/qocogen/actions/workflows/unit_tests.yml/badge.svg"><img src="https://github.com/qoco-org/qocogen/actions/workflows/unit_tests.yml/badge.svg"/></a>
  <a href="https://img.shields.io/pypi/dm/qocogen.svg?label=Pypi%20downloads"><img src="https://img.shields.io/pypi/dm/qocogen.svg?label=Pypi%20downloads" alt="PyPI Downloads" /></a>
  <a href="https://qoco-org.github.io/qoco/"><img src="https://img.shields.io/badge/docs-online-brightgreen?logo=read-the-docs&style=flat" alt="Documentation" /></a>
  <a href="https://opensource.org/licenses/BSD-3-Clause"><img src="https://img.shields.io/badge/License-BSD_3--Clause-green.svg" alt="License" /></a>
</p>

QOCOGEN is a code generator which takes in an second-order cone program problem family and generates a customized C solver (called qoco_custom) for the specified problem family which implements the same algorithm as QOCO. This customized solver is library-free, only uses static memory allocation, and can be a few times faster than QOCO.

## Installation and Usage

You can install `qocogen` by running `pip install qocogen`.

For instructions on using QOCOGEN, refer to the [documentation](https://qoco-org.github.io/qoco/codegen/index.html).

## Tests
To run tests, first install cvxpy and pytest
```bash
pip install cvxpy pytest
```

and execute:

```bash
pytest
```

## Bug reports

File any issues or bug reports using the [issue tracker](https://github.com/qoco-org/qocogen/issues).

## Citing
```
 @misc{chari2025qoco,
    author = {Chari, Govind M. and Acikmese, Behcet},
    title = {QOCO},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/qoco-org/qoco}},
 }
```

## License
QOCOGEN is licensed under the BSD-3-Clause license.
