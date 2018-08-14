# Convolved Negative Binomial Regression for Spatial Transcriptome Deconvolution
This repository contains code to perform Spatial Transcriptome Deconvolution with the Convolved Negative Binomial regression model described in<br>
*Charting Tissue Expression Anatomy by Spatial Transcriptome Deconvolution*<br>
Jonas Maaskola, Ludvig Bergenstråhle, Aleksandra Jurek, José Fernández Navarro, Jens Lagergren, Joakim Lundeberg<br>
doi: https://doi.org/10.1101/362624

## Dependencies
In order to compile it, you need the following dependencies:
* [Boost](http://www.boost.org/), version 1.58.0 or newer
* [Eigen](http://eigen.tuxfamily.org/), version 3
* [Flex](https://github.com/westes/flex)
* [Bison](https://www.gnu.org/software/bison/), version 3.0.4 or newer
* [LLVM](http://llvm.org/), version 5.0.0 or newer
  Please note that LLVM needs to be compiled with the runtime type identification (RTTI) feature enabled.
  This can be ensured by configuring LLVM with the following command:
```sh
cmake .. -DCMAKE_INSTALL_PREFIX=~/local/llvm -DCMAKE_BUILD_TYPE=RELWITHDEBINFO -DLLVM_BUILD_EXAMPLES=TRUE -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_ENABLE_RTTI=TRUE
```

Also, your C++ compiler should support OpenMP so that we can utilize parallel computation on multi-core systems.

## Compiling
You build and install the code as follows.
Note that ```<INSTALL_PREFIX>``` is a path below which the program will be installed.
This could be e.g. ```$HOME/local``` to install into a user-local prefix.

```sh
cd build
./gen_build.sh -DCMAKE_INSTALL_PREFIX=<INSTALL_PREFIX>
make
make install
```

The above will build both a release and a debug version of the code. Please use `make release` or `make debug` in place of `make` above if you want to build only the release or debug version. The binary for the release version will be called `std-nxt` and the binary for the debug version will be called `std-nxt-dbg`.

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
