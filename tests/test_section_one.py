import unittest

from gradescope_utils.autograder_utils.decorators import weight, number, partial_credit

import pandas as pd
import numpy as np

from utils import (
        extract_variables, 
        extract_initial_variables, 
        find_cells_with_text, 
        find_cells_by_indices,
        has_string_in_cell,
        has_string_in_code_cells,
        print_text_and_output_cells,
        print_code_and_output_cells)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.pipeline import Pipeline


class TestSectionOneTwo(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestSectionOneTwo, self).__init__(*args, **kwargs)
        self.notebook_path = None


    @weight(2.0)
    @number("1.1")
    def test_prepare_train_test_splits(self):
        print('')


        begin_cells = find_cells_with_text(self.notebook_path, "1.1. Preparation of Training and Testing Sets")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index'] - 1

        end_cells = find_cells_with_text(self.notebook_path, "1.2. Linear Regression: Fit a line for the given dataset")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)

        X_linear, Y_linear = cell_vars.get('X_linear', None), cell_vars.get('Y_linear', None)
        X_convex, Y_convex = cell_vars.get('X_convex', None), cell_vars.get('Y_convex', None)
        X_tri, Y_tri = cell_vars.get('X_tri', None), cell_vars.get('Y_tri', None)

        X_linear_train, X_linear_test, Y_linear_train, Y_linear_test = cell_vars.get('X_linear_train', None), cell_vars.get('X_linear_test', None), cell_vars.get('Y_linear_train', None), cell_vars.get('Y_linear_test', None)
        X_convex_train, X_convex_test, Y_convex_train, Y_convex_test = cell_vars.get('X_convex_train', None), cell_vars.get('X_convex_test', None), cell_vars.get('Y_convex_train', None), cell_vars.get('Y_convex_test', None)
        X_tri_train, X_tri_test, Y_tri_train, Y_tri_test = cell_vars.get('X_tri_train', None), cell_vars.get('X_tri_test', None), cell_vars.get('Y_tri_train', None), cell_vars.get('Y_tri_test', None)

        use_split_func = has_string_in_code_cells(self.notebook_path, begin_cell_idx, end_cell_idx, 'train_test_split')

        defined_all = \
            X_linear is not None and Y_linear is not None and \
            X_convex is not None and Y_convex is not None and \
            X_tri is not None and Y_tri is not None and \
            X_linear_train is not None and X_linear_test is not None and \
            Y_linear_train is not None and Y_linear_test is not None and \
            X_convex_train is not None and X_convex_test is not None and \
            Y_convex_train is not None and Y_convex_test is not None and \
            X_tri_train is not None and X_tri_test is not None and \
            Y_tri_train is not None and Y_tri_test is not None

        matching_length = \
            len(X_linear_train) + len(X_linear_test) == len(X_linear) and \
            len(Y_linear_train) + len(Y_linear_test) == len(Y_linear) and \
            len(X_convex_train) + len(X_convex_test) == len(X_convex)

        print("Defined all: ", defined_all)
        print("Matching length: ", matching_length)
        print("Use split func: ", use_split_func)

        result = defined_all and matching_length and use_split_func
        self.assertTrue(result);

    @weight(2.0)
    @number("1.2")
    def test_linear_regression(self):
        print('')

        self.assertTrue(True)

        begin_cells = find_cells_with_text(self.notebook_path, "1.2. Linear Regression: Fit a line for the given dataset")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']
        end_cells = find_cells_with_text(self.notebook_path, "1.3. Linear Model Evaluation (2 pts)")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)

        Y_linear_train_pred = cell_vars.get('Y_linear_train_pred', None)
        Y_convex_train_pred = cell_vars.get('Y_convex_train_pred', None)
        Y_tri_train_pred = cell_vars.get('Y_tri_train_pred', None)

        Y_linear_test_pred = cell_vars.get('Y_linear_test_pred', None)
        Y_convex_test_pred = cell_vars.get('Y_convex_test_pred', None)
        Y_tri_test_pred = cell_vars.get('Y_tri_test_pred', None)

        X_linear_train = cell_vars.get('X_linear_train', None)
        X_convex_train = cell_vars.get('X_convex_train', None)
        X_tri_train = cell_vars.get('X_tri_train', None)

        X_linear_test = cell_vars.get('X_linear_test', None)
        X_convex_test = cell_vars.get('X_convex_test', None)
        X_tri_test = cell_vars.get('X_tri_test', None)

        defined_all = \
                Y_linear_train_pred is not None and \
                Y_convex_train_pred is not None and \
                Y_tri_train_pred is not None and \
                Y_linear_test_pred is not None and \
                Y_convex_test_pred is not None and \
                Y_tri_test_pred is not None and \
                X_linear_train is not None and \
                X_convex_train is not None and \
                X_tri_train is not None

        matching_length = \
                len(Y_linear_train_pred) == len(X_linear_train) and \
                len(Y_convex_train_pred) == len(X_convex_train) and \
                len(Y_tri_train_pred) == len(X_tri_train) and \
                len(Y_linear_test_pred) == len(X_linear_test) and \
                len(Y_convex_test_pred) == len(X_convex_test) and \
                len(Y_tri_test_pred) == len(X_tri_test)

        use_fit = has_string_in_code_cells(self.notebook_path, begin_cell_idx, end_cell_idx, 'fit')
        use_predict = has_string_in_code_cells(self.notebook_path, begin_cell_idx, end_cell_idx, 'predict')

        print("Defined all: ", defined_all)
        print("Matching length: ", matching_length)
        print("Use fit: ", use_fit)
        print("Use predict: ", use_predict)

        result = defined_all and matching_length and use_fit and use_predict

        self.assertTrue(result)

    @partial_credit(2.0)
    @number("1.3")
    def test_linear_model_eval(self, set_score=None):
        print('')

        self.assertTrue(True)

        begin_cells = find_cells_with_text(self.notebook_path, "1.3. Linear Model Evaluation (2 pts)")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path, "1.4. Discussion about the Evaluation Results")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)

        mse_linear_train = cell_vars.get('mse_linear_train', None)
        mse_linear_test = cell_vars.get('mse_linear_test', None)
        r2_linear_train = cell_vars.get('r2_linear_train', None)
        r2_linear_test = cell_vars.get('r2_linear_test', None)

        mse_convex_train = cell_vars.get('mse_convex_train', None)
        mse_convex_test = cell_vars.get('mse_convex_test', None)
        r2_convex_train = cell_vars.get('r2_convex_train', None)
        r2_convex_test = cell_vars.get('r2_convex_test', None)

        mse_tri_train = cell_vars.get('mse_tri_train', None)
        mse_tri_test = cell_vars.get('mse_tri_test', None)
        r2_tri_train = cell_vars.get('r2_tri_train', None)
        r2_tri_test = cell_vars.get('r2_tri_test', None)

        defined_all = \
                mse_linear_train is not None and \
                mse_linear_test is not None and \
                r2_linear_train is not None and \
                r2_linear_test is not None and \
                mse_convex_train is not None and \
                mse_convex_test is not None and \
                r2_convex_train is not None and \
                r2_convex_test is not None and \
                mse_tri_train is not None and \
                mse_tri_test is not None and \
                r2_tri_train is not None and \
                r2_tri_test is not None

        has_r2_score = has_string_in_code_cells(self.notebook_path, begin_cell_idx, end_cell_idx, 'r2_score')
        has_mse = has_string_in_code_cells(self.notebook_path, begin_cell_idx, end_cell_idx, 'mean_squared_error')

        good_r2 = \
                r2_linear_train > 0.8 and r2_linear_test > 0.8 and \
                r2_convex_train < 0.4 and r2_convex_test < 0.4 and \
                r2_tri_train < 0.4 and r2_tri_test < 0.4

        print("Defined all: ", defined_all)
        print("Has r2_score: ", has_r2_score)
        print("Has mse: ", has_mse)
        print("Good r2: ", good_r2)


        if defined_all and has_r2_score and has_mse:
            if good_r2:
                set_score(2.0)
            else:
                set_score(1.0)
        else:
            set_score(0.0)

    @partial_credit(0.0)
    @number("1.4")
    def test_linear_model_eval_explanation(self, set_score=None):
        print('')
        begin_cells = find_cells_with_text(self.notebook_path, "1.4. Discussion about the Evaluation Results")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path, "1.5. Implement Polynomial Features")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        print_text_and_output_cells(self.notebook_path, begin_cell_idx, end_cell_idx)


    @partial_credit(2.0)
    @number("1.5")
    def test_implement_polynomial_features(self, set_score=None):
        print('')
        set_score(0.0)

        self.assertTrue(True)

        begin_cells = find_cells_with_text(self.notebook_path, "1.5. Implement Polynomial Features")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path, "1.6. Discussion about the Evaluation Results")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)

        Y_linear_train_pred_poly = cell_vars.get('Y_linear_train_pred_poly', None)
        Y_convex_train_pred_poly = cell_vars.get('Y_convex_train_pred_poly', None)
        Y_tri_train_pred_poly = cell_vars.get('Y_tri_train_pred_poly', None)

        Y_linear_test_pred_poly = cell_vars.get('Y_linear_test_pred_poly', None)
        Y_convex_test_pred_poly = cell_vars.get('Y_convex_test_pred_poly', None)
        Y_tri_test_pred_poly = cell_vars.get('Y_tri_test_pred_poly', None)

        has_predict = has_string_in_code_cells(self.notebook_path, begin_cell_idx, end_cell_idx, 'predict')
        has_fit = has_string_in_code_cells(self.notebook_path, begin_cell_idx, end_cell_idx, 'fit')

        done_pred_poly = \
                Y_linear_train_pred_poly is not None and \
                Y_convex_train_pred_poly is not None and \
                Y_tri_train_pred_poly is not None and \
                Y_linear_test_pred_poly is not None and \
                Y_convex_test_pred_poly is not None and \
                Y_tri_test_pred_poly is not None and \
                has_predict and has_fit


        mse_linear_test_poly = cell_vars.get('mse_linear_test_poly', None)
        r2_linear_test_poly = cell_vars.get('r2_linear_test_poly', None)

        mse_convex_test_poly = cell_vars.get('mse_convex_test_poly', None)
        r2_convex_test_poly = cell_vars.get('r2_convex_test_poly', None)

        mse_tri_test_poly = cell_vars.get('mse_tri_test_poly', None)
        r2_tri_test_poly = cell_vars.get('r2_tri_test_poly', None)

        mse_linear_train_poly = cell_vars.get('mse_linear_train_poly', None)
        r2_linear_train_poly = cell_vars.get('r2_linear_train_poly', None)

        mse_convex_train_poly = cell_vars.get('mse_convex_train_poly', None)
        r2_convex_train_poly = cell_vars.get('r2_convex_train_poly', None)

        mse_tri_train_poly = cell_vars.get('mse_tri_train_poly', None)
        r2_tri_train_poly = cell_vars.get('r2_tri_train_poly', None)

        done_metric = \
                mse_linear_test_poly is not None and \
                r2_linear_test_poly is not None and \
                mse_convex_test_poly is not None and \
                r2_convex_test_poly is not None and \
                mse_tri_test_poly is not None and \
                r2_tri_test_poly is not None and \
                mse_linear_train_poly is not None and \
                r2_linear_train_poly is not None and \
                mse_convex_train_poly is not None and \
                r2_convex_train_poly is not None and \
                mse_tri_train_poly is not None and \
                r2_tri_train_poly is not None

        good_metric = \
                r2_linear_train_poly > 0.8 and r2_linear_test_poly > 0.8 and \
                r2_convex_train_poly > 0.8 and r2_convex_test_poly > 0.8 and \
                r2_tri_train_poly > 0.6 and r2_tri_test_poly > 0.6


        score = 0.0
        if done_pred_poly:
            score += 1.0

        if done_metric:
            score += 0.5
            if good_metric:
                score += 0.5
        set_score(score)

    @partial_credit(0.0)
    @number("1.6")
    def test_poly_model_explanation(self, set_score=None):
        print('')
        begin_cells = find_cells_with_text(self.notebook_path, "1.6. Discussion about the Evaluation Results")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path, "1.7. Ridge/Lasso Regression (3 pts)")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        print_text_and_output_cells(self.notebook_path, begin_cell_idx, end_cell_idx)



    @partial_credit(3.0)
    @number("1.7")
    def test_ridge_lasso_regression(self, set_score=None):
        print('')

        self.assertTrue(True)

        begin_cells = find_cells_with_text(self.notebook_path, "1.6. Discussion about the Evaluation Results")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path, "Section 2")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)

        model_default = cell_vars.get('model_default', None)
        model_lasso = cell_vars.get('model_lasso', None)
        model_ridge = cell_vars.get('model_ridge', None)
        model_elastic = cell_vars.get('model_elastic', None)

        defined_all = \
                model_default is not None and \
                model_lasso is not None and \
                model_ridge is not None and \
                model_elastic is not None

        has_poly = False
        has_default = False
        has_lasso = False
        has_ridge = False
        has_elastic = False

        for model in [model_default, model_lasso, model_ridge, model_elastic]:
            if type(model) == Pipeline:
                for step in model.steps:
                    if(type(step[1]) == PolynomialFeatures):
                        has_poly = True
                    if(type(step[1]) == LinearRegression):
                        has_default = True
                    if(type(step[1]) == Lasso):
                        has_lasso = True
                    if(type(step[1]) == Ridge):
                        has_ridge = True
                    if(type(step[1]) == ElasticNet):
                        has_elastic = True
            else:
                if type(model) == LinearRegression:
                    has_default = True
                if type(model) == Lasso:
                    has_lasso = True
                if type(model) == Ridge:
                    has_ridge = True
                if type(model) == ElasticNet:
                    has_elastic = True

        # if more than three has_* are true, set good variable to true
        use_all_models = sum([has_default, has_lasso, has_ridge, has_elastic]) == 4
        use_most_models = sum([has_default, has_lasso, has_ridge, has_elastic]) >= 3
        use_poly_feat = has_poly

        print("Defined all: ", defined_all)
        print("Use proper models: ", use_all_models)
        print("Use poly feat: ", use_poly_feat)

        score = 0.0
        if defined_all:
            if use_all_models:
                score += 2.0
            elif use_most_models:
                score += 1.0

            if use_poly_feat:
                score += 1.0

        set_score(score)

    @partial_credit(0.0)
    @number("2.1")
    def test_gradient_derivation(self, set_score=None):
        print('')
        begin_cells = find_cells_with_text(self.notebook_path, "2.1. Derive the gradients")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path, "2.2 Implement our SimpleLinearRegression class")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        print_text_and_output_cells(self.notebook_path, begin_cell_idx, end_cell_idx)


    @partial_credit(8.0)
    @number("2.2")
    def test_implement_custom_model(self, set_score=None):
        print('')

        self.assertTrue(True)

        begin_cells = find_cells_with_text(self.notebook_path, "Section 2")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path, "2.3 Train your model")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)


        model = cell_vars.get('model', None)
        is_defined = model is not None

        SimpleLinearRegression = cell_vars.get('SimpleLinearRegression', None)
        is_complete = type(model) == SimpleLinearRegression

        results = cell_vars.get('results', None)
        answers = cell_vars.get('answers', None)

        good_answer = results == answers

        score = 0.0
        if is_defined:
            score += 3.0
            if is_complete:
                score += 3.0
                if good_answer:
                    score += 2.0

        set_score(score)

    @partial_credit(4.0)
    @number("2.3")
    def test_train_custom_model(self, set_score=None):
        print('')

        begin_cells = find_cells_with_text(self.notebook_path, "2.2 Implement our SimpleLinearRegression class")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path, "Section 3")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)


        model = cell_vars.get('model', None)
        is_defined = model is not None

        has_fit = has_string_in_code_cells(self.notebook_path, begin_cell_idx, end_cell_idx, 'fit')
        has_predict = has_string_in_code_cells(self.notebook_path, begin_cell_idx, end_cell_idx, 'predict')
    

        score = 0.0
        if is_defined:
            score += 1.0
            if has_fit:
                score += 1.5
                if has_predict:
                    score += 1.5

        set_score(score)

    @weight(1.0)
    @number("3.1")
    def test_theta_init(self):
        print('')

        begin_cells = find_cells_with_text(self.notebook_path, "Part 1 (1 point)")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']
        end_cells = find_cells_with_text(self.notebook_path, "Part 2 (3 points)")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)
        theta = cell_vars.get('theta', None)


        has_theta = theta is not None
        is_numpy_array = False
        theta_length_equals_two = False

        if has_theta:
            if type(theta) == np.ndarray:
                is_numpy_array = True
                if len(theta) == 2:
                    theta_length_equals_two = True

        print("Has theta: ", has_theta)
        print("Is numpy array: ", is_numpy_array)
        print("Theta length equals two: ", theta_length_equals_two)

        result = has_theta and is_numpy_array and theta_length_equals_two

        self.assertTrue(result)


    @weight(1.0)
    @number("3.3")
    def test_mse_loss_function(self):
        print('')

        begin_cells = find_cells_with_text(self.notebook_path, "Part 3: Write a function for the loss (1 point)")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']
        end_cells = find_cells_with_text(self.notebook_path, "Part 4: Training Loop (5 points)")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)
        mse_loss = cell_vars.get('mse_loss', None)


        has_mse = mse_loss is not None
        mse_correct = False

        gts = np.array([[1, 2, 3], [0.1, 0.3, 0.5], [10, 20, 30]])
        preds = np.array([[1, 2, 3], [0.05, 0.25, 0.55], [10, 25, 30]])

        mses = np.array([0, 0.0025, 8.3333333333333333333333])
        

        # Check whether mse_loss is working liks this
        # def mse_loss(true_values, predictions):
        #   return  np.mean((true_values- predictions)**2)
        if has_mse:
            if callable(mse_loss):
                count = 0
                for i in range(3):
                    if abs(np.array(mse_loss(gts[i], preds[i])).squeeze().item() - mses[i]) < 1e-4:
                        count += 1
                if count == 3:
                    mse_correct = True

        print("Has mse: ", has_mse)
        print("MSE correct: ", mse_correct)

        result = has_mse and mse_correct

        self.assertTrue(result)

    @partial_credit(0.0)
    @number("3.5")
    def test_part_3_5(self, set_score=None):
        print('')
        begin_cells = find_cells_with_text(self.notebook_path, "Part 5: What went wrong?")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path, "Handling More complicated data?")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        print_text_and_output_cells(self.notebook_path, begin_cell_idx, end_cell_idx)

    @partial_credit(0.0)
    @number("3.6")
    def test_part_3_6(self, set_score=None):
        print('')
        begin_cells = find_cells_with_text(self.notebook_path, "Part 6: List some different kernels you found online, their names.")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path, "this feature space, instead of guess and check with kernel methods, we invented")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        print_text_and_output_cells(self.notebook_path, begin_cell_idx, end_cell_idx)


    @partial_credit(0.0)
    @number("4.1")
    def test_part_bonus(self, set_score=None):
        print('')
        begin_cells = find_cells_with_text(self.notebook_path, "Bonus Question: Calculate the Exact Solution to the Linear Regressor")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        print_text_and_output_cells(self.notebook_path, begin_cell_idx, 999999)






