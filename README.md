# Edge detection server for ROCm

This repository contains a server as well as a request script for ROCm-accelerated edge detection from images. The intended use is to perform edge detection on the screen and determine if other images (templates) are contained in the processed image with the given similarity threshold.

## Usage

To use, create a config.py file that contains the following variables:

url_base - string of just the local host address ("x.x.x.x")  
url_port - int of just the port  
url - string (or an fstring based on earlier variables) with a local host address formated as "http://x.x.x.x:port/match"  

Launch the server with:  
python ./edge_server.py    
Launch the request with:  
python ./edge_request.py ./template_path/image.png  
where template_path is the path to whatever image is intended to be used as a template