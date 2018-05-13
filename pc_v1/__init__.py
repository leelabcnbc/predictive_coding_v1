"""a package that rewrites M W Spratling's various V1 predictive coding packages.

https://nms.kcl.ac.uk/michael.spratling/code.html
"""
import os
from sys import version_info

assert version_info >= (3, 6), 'must python 3.6 or higher'

root_dir = os.path.normpath(os.path.join(os.path.split(__file__)[0], '..'))
assert os.path.isabs(root_dir)

dir_dictionary = {
    'root': root_dir,
    'reference_V1_ResponseProperties': os.path.join(root_dir, 'reference', 'V1_ResponseProperties_debug'),
}
