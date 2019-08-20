# RFI_Monitor
A tool used for assisting observatory staff monitor Radio Frequency Interference around the site.

# Motivation
Many sources of man-made interference can be observed around the site, to which the telescopes are very sensitive to.
Previous attempts at interference monitoring have been limited to surveys covering a specific time range or instruments which have produced more data than could be digested by limited personnel resources. We want a system that can handle the volume of data produced and assist observatory staff in identifying and finding the source.

# To Install:
```bash
pip install ./
```

# Demonstration
See Video_Demo.ipynb for an example.

# Example
![image](https://drive.google.com/uc?export=view&id=1OzQYRGIY5hL4qodtgwFWxFCYtEdYNgNH)
10 minutes of data over 2.5 MHz. Observed at DRAO, Feb 2019.
Red Outlines are detection picked up by this tool.

# Requirements
numpy,
tensorflow,
sklearn,
skimage,
matplotlib

Built using python3.6.7

# Author
Rory Coles 

# Acknowledgments
RFI-Group at DRAO.
Code from Kyle Mills was used in tiling the images.
