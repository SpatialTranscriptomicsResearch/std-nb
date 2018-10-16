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

## How to use
Either: specify a number of paths to count matrix files in tabs-separated format (TSV), with genes in rows and spots in columns. There is a `--transpose` CLI switch if your matrices have spots in rows and genes in columns.

Or: specify a design file.
This is also tab separated and has to have at least one column named `path`, giving the paths to the files to be used. In addition, covariates can be annotated in this design file on a per-sample-basis.

The covariates specified in the design file can then be used in a model file, in which given columns `individual` and `treatment` in the design file could contain:
`rate = rate(gene) + rate(gene, type) + rate(type) + rate(type,spot) + rate(spot) + rate(gene, individual) + rate(gene, treatment)`
**Note**: while the line starts with `rate` this is actually the equation for the logarithm of the rate parameter of the NB!
In the model file you can also specify priors for coefficients that you introduce in the regression equations. But if you don't do that for a given coefficient it will be assumed to be standard normal distributed.
There is a simple utility program included, `std-spec-generator`, that helps with the process of writing a model file. Note that the following covariates are always pre-defined: `gene`, `section`, `spot`, `type`.

The most frequently used switches are:
`-v` / `--verbose`
`--design path`
`--model path`
`--transpose`
`-t N` for number of types
`--top N` to use only the highest expressed genes

A simple way to perform inference, not specifying any covariates, and using the auto-generated model `rate = rate(gene) + rate(gene, type) + rate(type) + rate(type,spot) + rate(spot) + rate(gene, section)` is done with the following command:
`std-nxt -t 20 matrix1.tsv matrix2.tsv -v`

Using your own covariates and model:
`std-nxt -t 20 --design design.txt --model model.txt -v`

The output basically consists of the gzipped TSV files for the scalars, vectors, and matrices, implied by the model, in the "covariate-..." files. And there are also two files of special interest: the expected counts in the gene-type and spot-type dimensions. Try visualizing the spot-type matrix's columns! I would recommend taking relative frequencies within the spots.

You can a little bit later experiment with the optimizer. By default, RPROP is used. You can use ADAM with `--optim adam`.

Hopefully this suffices to get you started! Have fun!

Related
=======
Note that you can find script to analyse the output of this package in [another git repository](https://gits-15.sys.kth.se/maaskola/multiScoopIBP-scripts).
