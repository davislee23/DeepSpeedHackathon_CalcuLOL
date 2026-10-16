"""Your team's driver. This package is the one you edit."""

import os
import sys

# The modules import each other flat (`import vehicle_model`) so the same files
# run as the ROS package, as standalone scripts and under the tests.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
