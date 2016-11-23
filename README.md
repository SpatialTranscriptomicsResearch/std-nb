This is code to perform Monte-Carlo Markov Chain (MCMC) inference for the
Spatial Transcriptome Deconvolution model.

In order to compile it, you need two libraries:
* [Boost](http://www.boost.org/)
* [Armadillo](http://arma.sourceforge.net/)

The Boost library needs to be at least version 1.58.0 because it was only in this version that the trigamma function (i.e. the curvature, or second derivative, of the gamma function) was introduced.

Compiling
=========
You build and install the code as follows.
Note that ```<INSTALL_PREFIX>``` is a path below which the program will be installed.
This could be e.g. ```$HOME/local``` to install into a user-local prefix.

```sh
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=<INSTALL_PREFIX>
make
make install
```

Note that ```<INSTALL_PREFIX>/bin``` has to be included in your ```$PATH``` variable.
To do this you have to have a line like

```sh
export PATH=<INSTALL_PREFIX>/bin:$PATH
```

in your ```$HOME/.bashrc``` file (or similar in case you are a shell other than bash).

Related
=======
Note that you can find script to analyse the output of this package in [another git repository](https://gits-15.sys.kth.se/maaskola/multiScoopIBP-scripts).
