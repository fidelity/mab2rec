Mab2Rec: Multi-Armed Bandits Recommender
========================================

Mab2Rec is a Python library for building bandit-based recommendation algorithms. It supports **context-free**, **parametric**
and **non-parametric** **contextual** bandit models powered by `MABWiser <https://github.com/fidelity/mabwiser>`_ and
fairness and recommenders evaluations powered by `Jurity <https://github.com/fidelity/jurity>`_.
It supports `all bandit policies available in MABWiser <https://github.com/fidelity/mabwiser#available-bandit-policies>`_.
The library is designed with rapid experimentation in mind, follows the `PEP-8 standards <https://www.python.org/dev/peps/pep-0008>`_ and is tested heavily.

Mab2Rec and several of the open-source software it is built on is developed by the Artificial Intelligence Center at Fidelity Investments, including:

* `MABWiser <https://github.com/fidelity/mabwiser>`_ to create multi-armed bandit recommendation algorithms (`IJAIT'21 <https://www.worldscientific.com/doi/abs/10.1142/S0218213021500214>`_, `ICTAI'19 <https://ieeexplore.ieee.org/document/8995418>`_).
* `TextWiser <https://github.com/fidelity/textwiser>`_ to create item representations via text featurization (`AAAI'21 <https://ojs.aaai.org/index.php/AAAI/article/view/17814>`_).
* `Selective <https://github.com/fidelity/selective>`_ to create user representations via feature selection.
* `Seq2Pat <https://github.com/fidelity/seq2pat>`_ to enhance users representations via sequential pattern mining (`AAAI'22 <https://aaai.org/Conferences/AAAI-22/>`_).
* `Jurity <https://github.com/fidelity/jurity>`_ to evaluate recommendations including fairness metrics (`ICMLA'21 <https://ieeexplore.ieee.org/abstract/document/9680169>`_).

An introduction to **content- and context-aware** recommender systems and an overview of the building blocks of the library is `presented at All Things Open 2021 <https://www.youtube.com/watch?v=54d_YUalvOA>`_.

.. include:: quick.rst

Source Code
===========
The source code is hosted on :repo:`GitHub <>`.

.. sidebar:: Contents

   .. toctree::
    :maxdepth: 2

    installation
    quick
    examples
    contributing
    api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
