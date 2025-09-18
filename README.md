# image-filters-zama

A fork of Zama's Hugging face space for the image filter application using Concrete ML FHE library

# Image filtering using FHE

## Run the application on your machine

In this directory, ie `image_filtering`, you can do the following steps.

### Install dependencies

First, create a virtual env and activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then, install required packages:

```bash
pip3 install pip --upgrade
pip3 install -U pip wheel setuptools --ignore-installed
pip3 install -r requirements.txt --ignore-installed
```

The above steps should only be done once.

## Run the app

In a terminal, run:

```bash
source .venv/bin/activate
python3 app.py
```

## Interact with the application

Open the given URL link (search for a line like `Running on local URL:  http://127.0.0.1:8888/`).

## Generate new filters

It is also possible to manually add some new filters in `filters.py`. Yet, in order to be able to use
them interactively in the app, you first need to update the `AVAILABLE_FILTERS` list found in `common.py`
and then compile them by running :

```bash
python3 generate_dev_filters.py
```

Check it finishes well (by printing "Done!").

## Comparison with Plaintext execution

|   Filter Name   | Ciphertext Time | Plaintext Time |     Ratio     |
| :-------------: | :-------------: | :------------: | :-----------: |
|    Identity     |   1.230688696   |  0.001823844   | 674.777391049 |
|     Inverse     |   1.408491823   |  0.000139474   | 10098.597753  |
|     Rotate      |   1.519377885   |  0.003917765   | 387.817514578 |
|    Grayscale    |   3.317794872   |  0.000217696   | 15240.4953329 |
|      Blur       |   2.841465296   |  0.057426847   | 49.4797371689 |
|     Sharpen     |   3.581822654   |  0.018348649   | 195.209067109 |
| Ridge Detection |   2.442341568   |  0.020646856   | 118.29120947  |
