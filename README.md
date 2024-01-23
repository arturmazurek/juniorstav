# Segment-Anything for sphericity

This is a sample python program giving an example of how open source can be utilised to build interactive applications that might be helpful for civil engineers in their field of science.
Specifically, it allows to select sand images from a picture and estimate their sphericity based on their found shape and a circle fitted to it.

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Dependencies

The following open source software is used to build this application:
* Meta's [Segment Anything](https://segment-anything.com) (SAM) model to find sand grains in a given picture.
* Circle fitting  library from [Project Nayuki](https://www.nayuki.io/page/smallest-enclosing-circle)
* [Matplotlib](https://matplotlib.org) for interactive graphing

## Installation

The application is written in python and so Python 3.x is a prerequisite. Install the latest one from [Python's website](https://www.python.org/downloads/).

Detailed intallation instructions for SAM and Matplotlib can be found on the respective project pages.
They can be installed e.g. by calling in terminal with Python installed:
```
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install matplotlib
```

Segment Anything to run additionally needs a model checkpoint. Once you download or clone this project from Github you will have to download the `default` model from SAM [Github page](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)
and put it into the main directory.

Finally, for simplicity smallest-enclosing-circle is put into this repository as there's no one-line method to install it.

## Usage

The application can be run by executing the following command from commandline:

```
python segmenter.py <path_to_image>
```
where `path_to_image` is a path to the sand image you want to open.

The application will display an interactive Matplotlib graph in which you can use left mouse button to mark sand grains and right mouse button to unmark the last marked grain. The cyan dot is the mouse click location and the bluish area is the sand shape found by SAM.

![image](https://github.com/arturmazurek/juniorstav/assets/2102059/1d0f17e3-8f84-4b65-a531-0e02001d33b4)

## Contributing

This project is intended as a presentation for civil engineers that it is possible with relative ease to utilise open source in their work. As such it's not intended for development. If you're interested in its functionality a published paper can be found at [TBD]
