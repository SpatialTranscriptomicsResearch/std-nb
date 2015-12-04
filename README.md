This is code to perform Metropolis-Hastings Monte-Carlo Markov Chain (MCMC) inference for the multi-scoop extension of the Indian Buffer Process (msIBP).

In order to compile it, your Boost version needs to be at least as recent as 1.58.0.
The reason why such a recent version of Boost is required is because it was only in this version that the trigamma function (i.e. the curvature, or second derivative, of the gamma function) was introduced.

Compiling
=========
You build and install the code as follows.
Note that ```<INSTALL_PREFIX>``` is a path below which the program will be installed.
This could be e.g. $HOME/local to install into a user-local prefix.

```sh
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=<INSTALL_PREFIX>
make
make install
```

Note that <INSTALL_PREFIX>/bin has to be included in your $PATH variable.
To do this you have to have a line like

```sh
export PATH=<INSTALL_PREFIX>/bin:$PATH
```

in your $HOME/.bashrc file (or similar in case you are a shell other than bash).
