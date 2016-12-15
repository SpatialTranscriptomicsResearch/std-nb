This is code to perform Monte-Carlo Markov Chain (MCMC) inference for the
Spatial Transcriptome Deconvolution model.

In order to compile it, you need two libraries:
* [Boost](http://www.boost.org/), version 1.58.0 or newer
* [Armadillo](http://arma.sourceforge.net/), version 6.400 or newer

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

Note that ```<INSTALL_PREFIX>/bin``` and ```<INSTALL_PREFIX>/lib``` have to be included in your ```PATH``` and ```LD_LIBRARY_PATH``` environment variables, respectively.

To do this you have to have lines like the following

```sh
export PATH=<INSTALL_PREFIX>/bin:$PATH
export LD_LIBRARY_PATH=<INSTALL_PREFIX>/lib:$LD_LIBRARY_PATH
```

to your ```$HOME/.bashrc``` file (or similar in case you are a shell other than bash).

Related
=======
Note that you can find script to analyse the output of this package in [another git repository](https://gits-15.sys.kth.se/maaskola/multiScoopIBP-scripts).
