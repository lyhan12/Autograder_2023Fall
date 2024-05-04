import unittest
from gradescope_utils.autograder_utils.decorators import weight
import os

class TestSubmission(unittest.TestCase):
    @weight(0)
    def test_ipynb_file_submitted(self):
        """Check if at least one .ipynb file is submitted"""
        # List all submitted files

        submission_dir = '/autograder/submission'
        # List everything in that directory.
        files_in_directory = os.listdir(submission_dir)
        # Filter the list to only .ipynb files.
        ipynb_files = [file for file in files_in_directory if file.endswith('.ipynb')]
        # Check that there is exactly one .ipynb file.
        self.assertEqual(len(ipynb_files), 1, f"Expected 1 .ipynb file, found {len(ipynb_files)}")
        # Check if there is at least one .ipynb file
