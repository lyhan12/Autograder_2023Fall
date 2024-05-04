import argparse
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

import unittest
from gradescope_utils.autograder_utils.json_test_runner import JSONTestRunner

import matplotlib  # Import matplotlib to set the backend
matplotlib.use('Agg')

from tests import test_section_one
from tests import test_files

import glob
import utils
import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default='/autograder/results/results.json')
    parser.add_argument('--notebook_path', type=str, default=None)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    debug_mode  = args.debug
    result_path = args.result_path
    notebook_path = args.notebook_path


    if notebook_path is None:


        submission_test = unittest.defaultTestLoader.loadTestsFromTestCase(test_files.TestSubmission)
        suite = unittest.TestSuite([submission_test])
        runner = unittest.TextTestRunner(verbosity=2)
        res = runner.run(suite)

        if not res.wasSuccessful():
            exit(0)


        notebook_files = glob.glob('/autograder/submission/*.ipynb')
        print(notebook_files)

        if len(notebook_files) == 0:
            print("Error: No notebook files found in /autograder/source/")
            # Handle error, maybe set a result status indicating failure
        elif len(notebook_files) > 1:
            print("Error: More than one notebook file found. Ensure only one notebook is submitted.")
            # Handle error, maybe set a result status indicating failure
        else:
            notebook_path = notebook_files[0]

        # Proceed with grading using notebook_path
        assert len(notebook_files) == 1






    suite = unittest.TestSuite()
    tests = unittest.defaultTestLoader.loadTestsFromTestCase(test_section_one.TestSectionOneTwo)
    for test in tests:
        setattr(test, 'notebook_path', notebook_path)
    suite.addTests(tests)

        
    with open(result_path, 'w') as f:
       runner = JSONTestRunner(visibility='visible', stream=f)
       runner.run(suite)


    # if debug_mode is True, read the json file and print test case with score
    if debug_mode:

        runner = unittest.TextTestRunner(verbosity=2)
        for test in tests:
            runner.run(test)

        print('')
        print("============ JSON Result ============")
        with open(result_path, 'r') as f:
            data = json.load(f)
            for test in data['tests']:
                print(f"{test['number']}: {test['score']}/{test['max_score']} ({test['name']})")

    



