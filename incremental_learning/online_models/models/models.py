#!/usr/bin/env python3

"""
Incremental Models Implementation

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : October 2022
Description : This contains the implementation of the online models using a
              singleton design pattern.

"""

import copy

import pickle
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
# mpl.rc('axes', labelsize=14)
# mpl.rc('xtick', labelsize=12)
# mpl.rc('ytick', labelsize=12)
# blue_color_matlab = (0, 0.4470, 0.7410)
import numpy as np

import river

from river import (
    metrics,
    #stream,
    #linear_model,
    #optim,
    #tree,
    preprocessing,
    forest,
    ensemble,
    neural_net,
    neighbors,
    rules,
    #model_selection,
    #conf,
    #stats,
    imblearn
)


class Singleton(type):
    """Metaclass to create classes with a Singleton design pattern."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Return the instance of the class if any, otherwise creates it."""

        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args,
                **kwargs
            )
        return cls._instances[cls]


class TrainingMonitor:
    """Training monitor for re-train and stop-training strategies management"""
    def __init__(self, model_type):

        # Operation mode attribute
        self.operation_mode = "train"
        # Variable storing old stage
        self.previous_stage = "train"
        # Variables used by the higher-level OnlineModels manager
        self.stage_changed = False

        # Training attributes
        self.train_current_iteration = 0
        self.train_start_training = True
        self.train_nominal_obs_btw_validation = 1000
        self.train_actual_obs_btw_validation = 1000
        self.train_obs_btw_validation_reduction_factor = 0
        self.train_training_metric = river.utils.Rolling(river.metrics.MAPE(), window_size=1000)
        self.train_training_metric_history = []  # TODO.Test
        self.train_validation_metric = 100.0
        self.train_validation_threshold = 1
        self.train_stable_training_error = 0
        self.train_stable_training_error_threshold = 2
        self.train_train_regions = []  # TODO: Remove. Test
        self.train_train_regions_color = "lightgreen"  # TODO: Remove. Test

        # Test attributes
        self.test_current_iteration = 0
        self.test_obs_btw_test = 2000
        self.test_nominal_obs_to_test = 200
        self.test_actual_obs_to_test = 200
        self.test_obs_to_test_reduction_factor = 0
        self.test_test_metric = river.metrics.MAPE()
        self.test_test_metric_history = []  # TODO. Remove test
        self.test_test_threshold = 2
        self.test_significant_error_variation = 0
        self.test_significant_error_variation_threshold = 2
        self.test_test_regions = []  # TODO: Remove. Test
        self.test_test_regions_color = "orange"  # TODO: Remove. Test

        # This will be hardcoded.
        # TODO: Needs to be replaced with the values that generates the best
        #       models, based on what the grid search evaluation says
        if model_type in ["PS", "PS+PL"]:
            self.train_obs_btw_validation_reduction_factor = 0
            self.train_validation_threshold = 1
            self.train_stable_training_error_threshold = 2
            self.test_test_threshold = 2
            self.test_obs_to_test_reduction_factor = 0
            self.test_significant_error_variation_threshold = 3
            print("PS model")
        elif model_type in ["PL"]:
            self.train_obs_btw_validation_reduction_factor = 0
            self.train_validation_threshold = 3
            self.train_stable_training_error_threshold = 2
            self.test_test_threshold = 3
            self.test_obs_to_test_reduction_factor = 0
            self.test_significant_error_variation_threshold = 3
            print("PL model")
        elif model_type in ["Time"]:
            self.train_obs_btw_validation_reduction_factor = 0
            self.train_validation_threshold = 3
            self.train_stable_training_error_threshold = 2
            self.test_test_threshold = 6
            self.test_obs_to_test_reduction_factor = 0
            self.test_significant_error_variation_threshold = 3
            print("Time model")
        else:
            raise ValueError(f"Model type '{model_type}' not recognized")


    def get_info(self):
        """Getter for accessing current training monitor info."""

        # Compute the minimum remaining training observations
        if self.operation_mode == "train":

            # Calculate the remaining training obs for this round
            actual_train_round_obs_left = \
                self.train_actual_obs_btw_validation - self.train_current_iteration

            # Add the following rounds
            train_rounds_left = \
                self.train_stable_training_error_threshold - self.train_stable_training_error - 1

            following_train_rounds_obs_left = 0
            # We need to add the trainig observations of the following
            # rounds taking into account the reduction factor
            for following_train_round_number in range(train_rounds_left):
                # - To take into account the reduction factor we compute the observations each round
                #   as the nominal value time (1 - n * reduction factor)
                # - This n value is the value of the
                #   (self.train_stable_training_error + 1 + follwing_round_number)
                #   ^ (self.train_stable_training_error + 1) is because we start computing
                #                                            from the round next to the actual one
                #   ^ (+ follwing_round_number) is because this loop compute all following rounds,
                #                               so in each iteration the reduction factor has to be
                #                               increased
                following_train_rounds_obs_left += round(self.train_nominal_obs_btw_validation * (1 - (self.train_stable_training_error + 1 + following_train_round_number) * self.train_obs_btw_validation_reduction_factor))

            # Add up actual round and following rounds remaining training obs
            minimum_train_obs_left = actual_train_round_obs_left + following_train_rounds_obs_left

            # Other info parameters (not related to train) are cleared
            minimum_test_idle_obs_left = 0
            minimum_test_test_round_obs_left = 0

        # Computations While models are trusted
        elif self.operation_mode == "idle":
            # Compute remaining obs where the models are trusted to be working properly
            minimum_test_idle_obs_left = self.test_obs_btw_test - self.test_current_iteration

            # Other info parameters (not related to test-idle) are cleared
            minimum_train_obs_left = 0
            minimum_test_test_round_obs_left = 0

        # While models have to be tested
        else:
            # Compute remaining test obs for this round
            # These are the minimum since other rounds are needed only when test error get worse
            minimum_test_test_round_obs_left = \
                self.test_actual_obs_to_test - self.test_current_iteration

            # Other info parameters (not related to test-test) are cleared
            minimum_train_obs_left = 0
            minimum_test_idle_obs_left = 0

        return {
            "operation_mode": self.operation_mode,
            "minimum_train_obs_left": minimum_train_obs_left,
            "minimum_test_idle_obs_left": minimum_test_idle_obs_left,
            "minimum_test_test_round_obs_left": minimum_test_test_round_obs_left
            }

    def update_idle_current_iteration(self, num_iterations):
        """
        Indicates the Training Monitor that 'num_iterations' idle observations
        have passed.
        """

        # TODO: Remove. For ploting the metrics
        self.train_training_metric_history += [0] * num_iterations

        # Check whether the test stage is operating in idle or testing mode
        if self.operation_mode == "idle":
            # In the idle mode we just wait for X observations where the
            # models are trusted to be properly train and after that we
            # perform a test phase
            # This process is repeated until the models are no longer well
            # trained and we go back to the training stage

            # Increment the current iteration counter
            self.test_current_iteration += num_iterations

            # Wait until X observations have elapsed
            if self.test_current_iteration == self.test_obs_btw_test:
                print("[Training Monitor]This shouldn't happen...")
                exit(1)
        else:
            print("[Training Monitor]Not in idle mode....")
            exit(1)

    def increment_idle_current_iteration(self):
        """Indicates the Training Monitor that one idle observations have passed."""

        # TODO: Remove. For ploting the metrics
        self.train_training_metric_history += [0]

        # Check whether the test stage is operating in idle or testing mode
        if self.operation_mode == "idle":
            # In the idle mode we just wait for X observations where the
            # models are trusted to be properly train and after that we
            # perform a test phase
            # This process is repeated until the models are no longer well
            # trained and we go back to the training stage

            # Increment the current iteration counter
            self.test_current_iteration += 1

            # Wait until X observations have elapsed
            if self.test_current_iteration == self.test_obs_btw_test:
                print("[Training Monitor]This shouldn't happen...")
                exit(1)
        else:
            print("[Training Monitor]Not in idle mode....")
            exit(1)

    # TODO: Remove iteration, is only for train/test regions ploting
    def end_idle(self, iteration):
        """
        Indicates the Training Monitor that the requiered idle observations
        have passed.
        """

        # TODO: Remove. For ploting the metrics
        num_iterations = self.test_obs_btw_test - self.test_current_iteration
        self.train_training_metric_history += [0] * num_iterations

        # Check whether the test stage is operating in idle or testing mode
        if self.operation_mode == "idle":
            # After those X observations the testing stage goes from
            # idle mode to test mode

            # Change operation mode to "test"
            self.operation_mode = "test"
            # Clear the current iteration counter for the next phase
            self.test_current_iteration = 0
            # Clear the test metric for the testing phase
            self.test_test_metric = river.metrics.MAPE()
            # Clear the significant error variation counter
            self.test_significant_error_variation = 0
            # TODO: Remove. Clear the metric history and mark the
            #       iteration of the beginning of the test process
            self.test_test_metric_history = []
            self.test_test_regions.append([iteration])
        else:
            print("[Training Monitor]Not in idle mode....")
            exit(1)

    def clear_stage_changed(self):
        """
        The self.stage_changed flag is used to signal other models to sync to
        the stage of the one with this flag activated.
        After the syncing process the flag should go back to False for not
        triggering more stage changes
        """

        if self.stage_changed:
            # When the stage_changed variable is set to True, clear it
            self.stage_changed = False
            return

        # When the variable is not True, you shouldn't be here...
        print(f"[Train Monitor] Cannot clear stage_changed since its value is {self.stage_changed}")
        exit()

    def reset_to_stage(self, new_stage):
        """
        Resets the state of the Training Monitor to a particular clean stage.
        It is used to sync the models.
        When any model jumps to a stage different than the others, those will
        be signaled to reset to the stage of the different one.
        This only apply to dominant stages (train > test > idle)
        """

        # Since we have been synced, flag indicating stage changed to 0
        self.stage_changed = False

        if new_stage == "train":
            # The model is going to be reset to the "train" stage
            if self.operation_mode == "train":
                # Task: Reset to "train" stage comming from same "train" stage
                # When: 1. All models are in "train" and any has to reset "train"
                #          so the others have to reset train also
                #
                # Actions: (same as going from "train" to "train" in .update())
                # 1. Clear train_current_iteration (implicitly done in update())
                # 2. Clear train_stable_training_error
                # 3. Set train_actual_obs_btw_validation as nominal
                # 4. Set train_validation_metric as train_training_metric

                # TODO: Remove. sanity check for development phase
                if self.previous_stage == "idle":
                    print("[reset_to_stage] When "
                          f"From: {self.operation_mode} -> "
                          f"To: {new_stage}. Previous stage cannot be: {self.previous_stage}"
                          )
                    exit()

                # Clear train_current_iteration
                self.train_current_iteration = 0
                # Clear train_stable_training_error
                self.train_stable_training_error = 0
                # Set train_actual_obs_btw_validation as nominal
                self.train_actual_obs_btw_validation = self.train_nominal_obs_btw_validation
                # Set train_validation_metric as train_training_metric
                self.train_validation_metric = self.train_training_metric.get()

            elif self.operation_mode == "idle":
                # Task: Reset to "train" stage comming from same "idle" stage
                # When: 1. All models are in "test" and some go idle by at least
                #          one goes to train so others have to go to train also
                #       2. Some models have gone from "train" to "idle" while
                #          at least one goes from "train" back to "re-train"

                # Find if it comes from "test" or "train"
                if self.previous_stage == "test":
                    # Task: Reset to "train" stage comming from same "idle" stage
                    # When: 1. All models are in "test" and some go idle by at least
                    #         one goes to train so others have to go to train also
                    #
                    # Actions: (1. Do from "test" to "train" in .update()
                    #           2. Remove what has already been done when
                    #              going from "test" to "idle" in .update()
                    #           3. Do from "idle" to "test" to clean the idle)
                    # 1. Set operation_mode to "train"
                    # 2. Set validation_metric as test_metric
                    # 3. Clear test_significant_error_variation
                    # 4. Clear test_current_iteration
                    # 5. Clear test_test_metric
                    # 6. Clear test_test_metric_history (TODO: Remove)

                    # Set operation_mode to "train"
                    self.operation_mode = "train"
                    # Set validation_metric as test_metric
                    self.train_validation_metric = self.test_test_metric.get()
                    # Clear test_significant_error_variation
                    self.test_significant_error_variation = 0
                    # Clear test_test_metric
                    self.test_current_iteration = 0
                    # Clear test_test_metric
                    self.test_test_metric = river.metrics.MAPE()
                    # Clear test_test_metric_history (TODO: Remove)
                    self.test_test_metric_history = []

                elif self.previous_stage == "train":
                    # Task: Reset to "train" stage comming from same "idle" stage
                    # When: 2. Some models have gone from "train" to "idle" while
                    #          at least one goes from "train" back to "re-train"
                    #
                    # Actions: (1. Do from "train" to "train" in .update()
                    #           2. Remove what has already been done when
                    #              going from "train" to "idle" in .update()
                    #              or revert it when unneeded (like train_region)
                    #           3. Do from "idle" to "test" to clean the idle)
                    # 1. Set operation_mode to "train"
                    # 2. Set train_validation_metric as train_training_metric
                    # 3. TODO: Revert train_start_training from True to False
                    # 4. TODO: Remove last train_train_regions element (was
                    #          indicating the end of the training stage, but we
                    #          go back to it)
                    # 5. Clear test_current_iteration

                    # Set operation_mode to "train"
                    self.operation_mode = "train"
                    # Set train_validation_metric as train_training_metric
                    self.train_validation_metric = self.train_training_metric.get()
                    # TODO: Revert train_start_training from True to False
                    self.train_start_training = False
                    # TODO: Remove last train_train_regions element (was
                    #       indicating the end of the training stage, but we
                    #       go back to it)
                    self.train_train_regions[-1].pop()
                    # Clear test_current_iteration
                    self.test_current_iteration = 0

            # The previous state cannot be any other when you are signaled
            # to go back to "train"
            else:
                print(
                    f"[reset_to_stage] Impossible. From: {self.operation_mode} -> To: {new_stage}"
                )
                exit()

        # The model is going to be reset to the "idle" stage
        elif new_stage == "idle":
            # When restarting from the same "idle" stage
            # This happens when the models are in "test" and while some have
            # an acceptable error and go to idle, at least one has to go test
            # again. The others are let to remain in the idle but.
            # If the model that was at test goes to train all will go train
            # If the model that was at test goes idle all will reset to idle
            if self.operation_mode == "idle":
                # Task: Reset to "idle" stage comming from same "idle" stage
                # When: 1. All models are in "test" and some goes to "idle" while
                #          at least one of the goes back to "test".
                #
                #     ***********************************************************
                #       What we do is let all the models that go to "idle" to
                #       be there while the other model is in "test".
                #         1. In case that model goes then to "train" the models
                #            in "idle" will be signaled to reset from "idle" to
                #            "train" (this is handle in previous branches).
                #         2. In case that model goes then to "idle" the models
                #            already in "idle" will be signaled to reset from
                #            "idle" again back to a fresh "idle" stage.
                #     ***********************************************************
                #
                # Actions: (in "idle" only test_current_iteration is modified)
                # 1. Clear train_current_iteration

                # Clear train_current_iteration
                self.test_current_iteration = 0

            # The previous state cannot be any other when you are signaled
            # to go back to "idle"
            else:
                print(
                    f"[reset_to_stage] Impossible. From: {self.operation_mode} -> To: {new_stage}"
                )
                exit()

        # When test (which is a stage that is not restarted) or any other stage
        # it means there has been a bug, so exit
        else:
            print(f"'{new_stage}' is not a possible stage to reset to.")
            exit()

    def get_state(self):
        """Return the actual state of the Training Monitor"""

        return self.operation_mode, self.stage_changed

    # TODO: Remove iteration, is only for train/test regions ploting
    def update(self, y, y_pred, iteration):
        """
        Updates the Training Monitor with the current observations.
        Returns whether to train or not, as well as a signal indicating a
        stage change that should force other models to same stage.
        This signal will be used on a higher level to sync the models.
        """

        # Flag indicating if there is a change in the stage that should trigger
        # stage changes in the other models
        self.stage_changed = False

        # Update the training metric
        self.train_training_metric = self.train_training_metric.update(y, y_pred)
        # TODO: Remove
        self.train_training_metric_history.append(self.train_training_metric.get())

        # Training stage tasks
        # - Train for X observations Y consecutive times
        # - Go back to train if the error metric remains stable during the
        #   process
        # - Restart the training stage if the metric does not remain stable
        if self.operation_mode == "train":
            # Increment the current_iteration counter
            self.train_current_iteration += 1

            # TODO: Remove. Mark the iteration of the beginning of the training process
            if self.train_start_training:
                self.train_start_training = False
                # TODO: Remove
                self.train_train_regions.append([iteration])

            # Wait until x observations have elapsed
            if self.train_current_iteration == self.train_actual_obs_btw_validation:
                # The current iteration counter is cleared since we are going
                # to count again X observation (either in a subsequent train
                # phase or in a future one after a testing phase)
                self.train_current_iteration = 0

                # Craft the train condition (cleaner that directly in the if statement)
                # It check if the train metric is within a train_validation_threshold range
                # True:  within range (one step closer to stop training)
                # False: out of range (reset the trainign stage)
                tmp_train_condition = (
                    (self.train_training_metric.get() - self.train_validation_threshold)
                    <= self.train_validation_metric
                    <= (self.train_training_metric.get() + self.train_validation_threshold)
                )

                # Check if the train condition passes
                if tmp_train_condition:
                    # If the code reaches this point it means the training
                    # error is within the validation threshold after X
                    # observations

                    # Increment stable_training_error counter bc validation_metric is within range
                    self.train_stable_training_error += 1
                    # Reduce the actual observations between validations by a factor
                    self.train_actual_obs_btw_validation = round(self.train_nominal_obs_btw_validation * (1 - self.train_stable_training_error * self.train_obs_btw_validation_reduction_factor))
                else:
                    # From "train" to "train"
                    # If the code reaches this point it means the training
                    # error is not within the validation threshold after X
                    # observations

                    # Set previous stage as "train"
                    self.previous_stage = "train"

                    # Clear the stable_training_error counter
                    self.train_stable_training_error = 0
                    # Set the actual observations between validations to its nominal
                    self.train_actual_obs_btw_validation = self.train_nominal_obs_btw_validation
                    # Freeze the training_metric. Storing it in validation_metric
                    self.train_validation_metric = self.train_training_metric.get()
                    # Signal a change in the stage to other models
                    self.stage_changed = True

            # Stop training and go to testing mode when the stable training
            # error counter reaches the threshold indicating the model is well
            # trained
            if self.train_stable_training_error == self.train_stable_training_error_threshold:

                # Set previous stage as "train"
                self.previous_stage = "train"

                # Change operation model to "idle"
                self.operation_mode = "idle"
                # TODO: Remove. Mark start_training as True for next training phase
                self.train_start_training = True
                # Clear the stable training counter
                self.train_stable_training_error = 0
                # Set the actual obs between validations to its nominal
                self.train_actual_obs_btw_validation = self.train_nominal_obs_btw_validation
                # TODO: Remove. Mark the iteration of the end of the training process
                self.train_train_regions[-1].append(iteration)

        # Test stage tasks
        # - Wait in idle mode for X observations (without even predicting)
        # - After the X observations perform predictions for m observations n
        #   consecutive times
        # - If the testing metric remains stable during the process go back to
        #   the idle mode and repeat the process
        # - If there are importan variations in the testing metric start go
        #   back to the training stage

        # Check whether the test stage is operating in idle or testing mode
        elif self.operation_mode == "idle":
            # In the idle mode we just wait for X observations where the
            # models are trusted to be properly train and after that we
            # perform a test phase
            # This process is repeated until the models are no longer well
            # trained and we go back to the training stage

            # Increment the current iteration counter
            self.test_current_iteration += 1

            # Wait until X observations have elapsed
            if self.test_current_iteration == self.test_obs_btw_test:
                # After those X observations the testing stage goes from
                # idle mode to test mode

                # Set previous stage as "idle"
                self.previous_stage = "idle"

                # Change operation mode to "test"
                self.operation_mode = "test"
                # Clear the current iteration counter for the next phase
                self.test_current_iteration = 0
                # Clear the test metric for the testing phase
                self.test_test_metric = river.metrics.MAPE()
                # Clear the significant error variation counter
                self.test_significant_error_variation = 0
                # TODO: Remove. Clear the metric history and mark the
                #       iteration of the beginning of the test process
                self.test_test_metric_history = []
                self.test_test_regions.append([iteration])

        # Test mode
        else:
            # In the "test" mode we make m predictions n consecutive times
            # If the test metrics remain stable we go back to "idle" mode,
            # otherwise we go back to the training stage

            # Increment the current iteration counter
            self.test_current_iteration += 1

            # Predict with the adaptative model
            # TODO: place the prediction here!!

            # Update the test metric
            self.test_test_metric.update(y, y_pred)
            # TODO: Remove
            self.test_test_metric_history.append(self.test_test_metric.get())

            # Wait until m observations have elapsed
            if self.test_current_iteration == self.test_actual_obs_to_test:

                # The current iteration counter is cleared since we are
                # going to count again m observation (either in a
                # subsequent test phase or in a future one after a training
                # phase)
                self.test_current_iteration = 0

                # Craft the test condition (cleaner that directly in the if statement)
                # It check if the test metric is withing a test_threshold range
                # True:  within range (go back to "idle" again)
                # False: out of range (one step closer to re-start train)
                tmp_test_condition = (
                    (self.train_validation_metric - self.test_test_threshold)
                    <= self.test_test_metric.get()
                    <= (self.train_validation_metric + self.test_test_threshold)
                )

                # Check if the test condition passes
                if tmp_test_condition is False:
                    # From "test" to "test"
                    # If the code reaches this point it means the testing
                    # error is not within the test_threshold range after
                    # m observations

                    # Set previous stage as "test"
                    self.previous_stage = "test"

                    # Increment the significan error variation counter bc
                    # the test_metric is not within range
                    self.test_significant_error_variation += 1

                    # Reduce the actual observations to test by a factor
                    self.test_actual_obs_to_test = round(self.test_nominal_obs_to_test * (1 - self.test_significant_error_variation * self.test_obs_to_test_reduction_factor))

                else:
                    # From "test" to "idle"
                    # If the code reaches this point it means the testing
                    # error is within the test_threshold range after
                    # m observations

                    # Set previous stage as "test"
                    self.previous_stage = "test"

                    # Set the test operation mode back to "idle"
                    self.operation_mode = "idle"

                    # Set back the actual observations to test to its
                    # nominal value
                    self.test_actual_obs_to_test = self.test_nominal_obs_to_test
                    # TODO: Remove. Mark the iteration of the end of the test process
                    self.test_test_regions[-1].append(iteration)
                    # Signal a change in the stage to other models
                    self.stage_changed = True

            # Go to train when significant_error_variation reaches a
            # threshold indicating the model is not performing properly
            if (
                self.test_significant_error_variation
                == self.test_significant_error_variation_threshold
                ):

                # Set previous stage as "test"
                self.previous_stage = "test"

                # Set operation mode to "train"
                self.operation_mode = "train"
                # Store the actual training metric on the validation metric
                self.train_validation_metric = self.test_test_metric.get()
                # Clear the test metric and (# TODO) history
                self.test_test_metric = river.metrics.MAPE()
                self.test_test_metric_history = []
                # Clear the significant error variation counter
                self.test_significant_error_variation = 0
                # Set the actual obs to test back to its nominal value
                self.test_actual_obs_to_test = self.test_nominal_obs_to_test
                # TODO: Remove. Mark the iteration of the end of the test process
                self.test_test_regions[-1].append(iteration)
                # Signal a change in the stage to other models
                self.stage_changed = True

        # TODO: Remove. just for debuggin.
        # Return whether the model needs to learn from the actual observation
        return self.operation_mode, self.stage_changed


class Model():
    """Online model implementation (parent class)."""
    def __init__(self, model, metric, model_type):
        self._model = model
        self._metric = metric
        self._type = model_type
        self._training_monitor = TrainingMonitor(self._type)

    def update_state(self, y, y_pred, iteration):
        """
        Updates the Training Monitor with the current observations.
        Returns whether to train or not, as well as a signal indicating a
        stage change that should force other models to same stage.
        This signal will be used on a higher level to sync the models.
        """
        return self._training_monitor.update(y, y_pred, iteration)


    def get_info(self):
        """Getter for accessing current training monitor info."""
        return self._training_monitor.get_info()

    def get_state(self):
        """Return the actual state of the Training Monitor"""
        return self._training_monitor.get_state()

    def end_idle_phase(self, iteration):
        """
        Indicates the Training Monitor that the requiered idle observations
        have passed.
        """
        self._training_monitor.end_idle(iteration)

    def update_idle_phase(self, num_iterations):
        """
        Indicates the Training Monitor that 'num_iterations' idle observations
        have passed.
        """
        self._training_monitor.update_idle_current_iteration(num_iterations)

    def increment_idle_phase(self):
        """Indicates the Training Monitor that one idle observations have passed."""
        self._training_monitor.increment_idle_current_iteration()

    def clear_stage_changed_flag(self):
        """
        The self.stage_changed flag is used to signal other models to sync to
        the stage of the one with this flag activated.
        After the syncing process the flag should go back to False for not
        triggering more stage changes
        """
        self._training_monitor.clear_stage_changed()

    def reset_to_stage(self, new_stage):
        """
        Resets the state of the Training Monitor to a particular clean stage.
        It is used to sync the models.
        When any model jumps to a stage different than the others, those will
        be signaled to reset to the stage of the different one.
        This only apply to dominant stages (train > test > idle)
        """
        self._training_monitor.reset_to_stage(new_stage)

    def grid_search_train_batch(self, features_df, labels_df, grid_search_parameters):
        """Learn all observations within the 'feature_df' dataframe
           one by one.
        """

        i = 0

        # Adaptative Model Testing
        adaptative_model_data = {}
        adaptative_model_data["model"] = copy.deepcopy(self._model)     # Copia del modelo
        adaptative_model_data["operation_mode"] = "train"       # train or test
        adaptative_model_data["train"] = {}
        adaptative_model_data["train"]["current_iteration"] = 0
        # adaptative_model_data["train"]["nominal_obs_btw_validation"] = 1000             # Numero inicial de observaciones a entrenar antes de comprobar la validation_metric
        # adaptative_model_data["train"]["obs_btw_validation_reduction_factor"] = 0.2  # Factor por el que se reduce el numero de observaciones entre validaciones si en la anterior validación no se supera threshold
        adaptative_model_data["train"]["training_metric"] = river.utils.Rolling(river.metrics.MAPE(), window_size=1000)
        adaptative_model_data["train"]["training_metric_history"] = []
        adaptative_model_data["train"]["validation_metric"] = 100.0              # Metrica de validación empleada para comprobar si el modelo está suficientemente entrenado
        # adaptative_model_data["train"]["validation_threshold"] = 2.5               # Validation_error_threshold. Puntos porcentuales, para MAPE
        adaptative_model_data["train"]["stable_training_error"] = 0              # Contador que indica el numero de veces consecutivas que se supera "validation_threshold"
        # adaptative_model_data["train"]["stable_training_error_threshold"] = 4    # stable_training_error counter threshold. Habrá que ver cuál cuadra
        adaptative_model_data["train"]["train_regions"] = []                     # this list contains tuples indicatingn the training regions of the modes [(t0start, t0end), ..., (tnstart, tnend)]
        adaptative_model_data["train"]["train_regions_color"] = "lightgreen"
        adaptative_model_data["train"]["start_training"] = True  # This flag is used to know if this is the beginning of the train process to store the iteration number on the train_regions variable for posterior plotting
        adaptative_model_data["test"] = {}
        adaptative_model_data["test"]["operation_mode"] = "idle"        # idle or test
        adaptative_model_data["test"]["current_iteration"] = 0
        # adaptative_model_data["test"]["obs_btw_test"] = 1000             # Numero de observaciones a entrenar antes de comprobar la validation_metric
        # adaptative_model_data["test"]["nominal_obs_to_test"] = 200              # Numero inicial de observaciones a entrenar antes de comprobar la validation_metric
        # adaptative_model_data["test"]["obs_to_test_reduction_factor"] = 0.2  # Factor por el que se reduce el numero de observaciones a testear si en el anterior testeo se supera threshold
        adaptative_model_data["test"]["test_metric"] = river.metrics.MAPE()
        adaptative_model_data["test"]["test_metric_history"] = []
        # adaptative_model_data["test"]["test_threshold"] = 2.5               # Validation_error_threshold. Puntos porcentuales, para MAPE
        adaptative_model_data["test"]["significant_error_variation"] = 0              # Contador que indica el numero de veces consecutivas que se supera "validation_threshold"
        # adaptative_model_data["test"]["significant_error_variation_threshold"] = 2    # stable_training_error counter threshold. Habrá que ver cuál cuadraadaptative_model_data["train"]["train_regions"] = []                     # this list contains tuples indicaten the training regions of the modes [(t0start, t0end), ..., (tnstart, tnend)]
        adaptative_model_data["test"]["test_regions"] = []                     # this list contains tuples indicatingn the test in test mode regions of the modes [(t0start, t0end), ..., (tnstart, tnend)]
        adaptative_model_data["test"]["test_regions_color"] = "orange"
        adaptative_model_data["frozen"] = {}
        adaptative_model_data["frozen"]["model"] = copy.deepcopy(self._model)
        adaptative_model_data["frozen"]["training_metric"] = river.utils.Rolling(river.metrics.MAPE(), window_size=1000)
        adaptative_model_data["frozen"]["training_metric_history"] = []
        adaptative_model_data["frozen"]["froze_flag"] = False

        # Get parameters for grid search
        # Train
        adaptative_model_data["train"]["nominal_obs_btw_validation"] = grid_search_parameters["parameters"]["train"]["nominal_obs_btw_validation"]
        adaptative_model_data["train"]["obs_btw_validation_reduction_factor"] = grid_search_parameters["parameters"]["train"]["obs_btw_validation_reduction_factor"]
        adaptative_model_data["train"]["validation_threshold"] = grid_search_parameters["parameters"]["train"]["validation_threshold"]
        adaptative_model_data["train"]["stable_training_error_threshold"] = grid_search_parameters["parameters"]["train"]["stable_training_error_threshold"]
        # Test
        adaptative_model_data["test"]["obs_btw_test"] = grid_search_parameters["parameters"]["test"]["obs_btw_test"]
        adaptative_model_data["test"]["nominal_obs_to_test"] = grid_search_parameters["parameters"]["test"]["nominal_obs_to_test"]
        adaptative_model_data["test"]["test_threshold"] = grid_search_parameters["parameters"]["test"]["test_threshold"]
        adaptative_model_data["test"]["obs_to_test_reduction_factor"] = grid_search_parameters["parameters"]["test"]["obs_to_test_reduction_factor"]
        adaptative_model_data["test"]["significant_error_variation_threshold"] = grid_search_parameters["parameters"]["test"]["significant_error_variation_threshold"]

        # Print model type
        print(self._type)

        # List for storing continuously trained model metrics
        continuous_train_mape_history = []

        for x, y in river.stream.iter_pandas(features_df, labels_df, shuffle=False, seed=42):

            # Cast variables
            # Check if there is cpu_usage data in the observations
            if "user" in x:
                x["user"] = float(x["user"])
                x["kernel"] = float(x["kernel"])
                x["idle"] = float(x["idle"])

            x["Main"] = int(x["Main"])
            x["aes"] = int(x["aes"])
            x["bulk"] = int(x["bulk"])
            x["crs"] = int(x["crs"])
            x["kmp"] = int(x["kmp"])
            x["knn"] = int(x["knn"])
            x["merge"] = int(x["merge"])
            x["nw"] = int(x["nw"])
            x["queue"] = int(x["queue"])
            x["stencil2d"] = int(x["stencil2d"])
            x["stencil3d"] = int(x["stencil3d"])
            x["strided"] = int(x["strided"])
            y = float(y)

            #############################
            ## Test: modelo adaptativo ##
            #############################

            # Predict with the adaptative model
            adaptative_model_y_pred = adaptative_model_data["model"].predict_one(x)
            adaptative_model_data["train"]["training_metric"] = adaptative_model_data["train"]["training_metric"].update(y, adaptative_model_y_pred)
            adaptative_model_data["train"]["training_metric_history"].append(adaptative_model_data["train"]["training_metric"].get())

            # Frozen model
            adaptative_model_y_pred_frozen = adaptative_model_data["frozen"]["model"].predict_one(x)
            adaptative_model_data["frozen"]["training_metric"] = adaptative_model_data["frozen"]["training_metric"].update(y, adaptative_model_y_pred_frozen)
            adaptative_model_data["frozen"]["training_metric_history"].append(adaptative_model_data["frozen"]["training_metric"].get())

            # Only when in training mode
            if adaptative_model_data["operation_mode"] == "train":
                if adaptative_model_data["train"]["start_training"]:
                    # Mark the iteration of the beginning of the training process
                    adaptative_model_data["train"]["train_regions"].append([i])
                    # Set the actual_obs_btw_validation to its nominal value
                    tmp_train_actual_obs_btw_validation = adaptative_model_data["train"]["nominal_obs_btw_validation"]
                    adaptative_model_data["train"]["start_training"] = False

                adaptative_model_data["model"].learn_one(x, y)
                adaptative_model_data["train"]["current_iteration"] += 1


                # Entrenamos el modelo congelado hasta que se congele
                if adaptative_model_data["frozen"]["froze_flag"] is False:
                    adaptative_model_data["frozen"]["model"].learn_one(x, y)

                # Wait until x observations elapse
                if adaptative_model_data["train"]["current_iteration"] == tmp_train_actual_obs_btw_validation:
                    # Check whether the validation_metric is within the training_metric +- validation_threshold
                    if (adaptative_model_data["train"]["training_metric"].get() - adaptative_model_data["train"]["validation_threshold"]) <= adaptative_model_data["train"]["validation_metric"] <= (adaptative_model_data["train"]["training_metric"].get() + adaptative_model_data["train"]["validation_threshold"]):

                        adaptative_model_data["train"]["current_iteration"] = 0
                        # Increment the stable_training_error  counter if the validation_metric is within range
                        adaptative_model_data["train"]["stable_training_error"] += 1
                        tmp_train_actual_obs_btw_validation = round(adaptative_model_data["train"]["nominal_obs_btw_validation"] * (1 - adaptative_model_data["train"]["stable_training_error"] * adaptative_model_data["train"]["obs_btw_validation_reduction_factor"]))
                    else:

                        adaptative_model_data["train"]["current_iteration"] = 0
                        # Clear the stable_training_error counter and freeze the training_metric inside the validation_metric if the validation_metric is outside range
                        adaptative_model_data["train"]["stable_training_error"] = 0
                        tmp_train_actual_obs_btw_validation = adaptative_model_data["train"]["nominal_obs_btw_validation"]
                        adaptative_model_data["train"]["validation_metric"] = adaptative_model_data["train"]["training_metric"].get()

                # Stop training when the stable_training_error counter reaches the threshold indicating the model is well trained
                if adaptative_model_data["train"]["stable_training_error"] == adaptative_model_data["train"]["stable_training_error_threshold"]:
                    adaptative_model_data["operation_mode"] = "test"
                    adaptative_model_data["train"]["start_training"] = True
                    adaptative_model_data["train"]["current_iteration"] = 0
                    adaptative_model_data["train"]["stable_training_error"] = 0
                    tmp_train_actual_obs_btw_validation = adaptative_model_data["train"]["nominal_obs_btw_validation"]
                    # Mark the iteration of the end of the training process
                    adaptative_model_data["train"]["train_regions"][-1].append(i)

                    # Congelamos para siempre
                    adaptative_model_data["frozen"]["froze_flag"] = True

            else:
                if adaptative_model_data["test"]["operation_mode"] == "idle":

                    adaptative_model_data["test"]["current_iteration"] += 1

                    if adaptative_model_data["test"]["current_iteration"] == adaptative_model_data["test"]["obs_btw_test"]:

                        adaptative_model_data["test"]["operation_mode"] = "test"
                        adaptative_model_data["test"]["current_iteration"] = 0
                        adaptative_model_data["test"]["test_metric"] = river.metrics.MAPE()
                        adaptative_model_data["test"]["test_metric_history"] = []
                        adaptative_model_data["test"]["significant_error_variation"] = 0
                        # Mark the iteration of the beginning of the test process
                        adaptative_model_data["test"]["test_regions"].append([i])
                        # Set the actual_obs_to_test to its nominal value
                        tmp_test_actual_obs_to_test = adaptative_model_data["test"]["nominal_obs_to_test"]

                else:

                    adaptative_model_data["test"]["current_iteration"] += 1

                    # Predict with the adaptative model
                    adaptative_model_data["test"]["test_metric"].update(y, adaptative_model_y_pred)
                    adaptative_model_data["test"]["test_metric_history"].append(adaptative_model_data["test"]["test_metric"].get())

                    if adaptative_model_data["test"]["current_iteration"] == tmp_test_actual_obs_to_test:

                        tmp_test_condition = ( (adaptative_model_data["train"]["validation_metric"] - adaptative_model_data["test"]["test_threshold"]) <= adaptative_model_data["test"]["test_metric"].get() <= (adaptative_model_data["train"]["validation_metric"] + adaptative_model_data["test"]["test_threshold"]) )

                        if tmp_test_condition is False:
                            adaptative_model_data["test"]["current_iteration"] = 0
                            adaptative_model_data["test"]["significant_error_variation"] += 1
                            tmp_test_actual_obs_to_test = round(adaptative_model_data["test"]["nominal_obs_to_test"] * (1 - adaptative_model_data["test"]["significant_error_variation"] * adaptative_model_data["test"]["obs_to_test_reduction_factor"]))
                        else:
                            adaptative_model_data["test"]["current_iteration"] = 0
                            adaptative_model_data["test"]["operation_mode"] = "idle"
                            adaptative_model_data["test"]["current_iteration"] = 0
                            tmp_test_actual_obs_to_test = adaptative_model_data["test"]["nominal_obs_to_test"]
                            # Mark the iteration of the end of the test process
                            adaptative_model_data["test"]["test_regions"][-1].append(i)

                    # Go to train when significant_error_variation reaches a threshold indicating the model is not performing properly
                    if adaptative_model_data["test"]["significant_error_variation"] == adaptative_model_data["test"]["significant_error_variation_threshold"]:
                        adaptative_model_data["operation_mode"] = "train"
                        adaptative_model_data["test"]["operation_mode"] = "idle"
                        adaptative_model_data["test"]["current_iteration"] = 0
                        adaptative_model_data["test"]["test_metric"] = river.metrics.MAPE()
                        adaptative_model_data["test"]["test_metric_history"] = []
                        adaptative_model_data["test"]["significant_error_variation"] = 0
                        tmp_test_actual_obs_to_test = adaptative_model_data["test"]["nominal_obs_to_test"]
                        adaptative_model_data["train"]["validation_metric"] = adaptative_model_data["train"]["training_metric"].get()
                        # Mark the iteration of the end of the test process
                        adaptative_model_data["test"]["test_regions"][-1].append(i)

            #print("[Adaptative model - train] train?: {} | cur_iter: {} | train_metric: {} | validation_metric: {} | stable_training_error: {} | stable_training_error_threshold: {}".format(adaptative_model_data["operation_mode"] == "train", adaptative_model_data["train"]["current_iteration"], adaptative_model_data["train"]["training_metric"].get(), adaptative_model_data["train"]["validation_metric"], adaptative_model_data["train"]["stable_training_error"], adaptative_model_data["train"]["stable_training_error_threshold"]))
            #print("[Adaptative model - test]  test?: {} | mode: {} | cur_iter: {} | test_metric: {} | validation_metric: {} | sign_error_variation: {} | sign_error_variation_threshold: {}".format(adaptative_model_data["operation_mode"] == "test", adaptative_model_data["test"]["operation_mode"], adaptative_model_data["test"]["current_iteration"], adaptative_model_data["test"]["test_metric"].get(), adaptative_model_data["train"]["validation_metric"], adaptative_model_data["test"]["significant_error_variation"], adaptative_model_data["test"]["significant_error_variation_threshold"]))

            # Make prediction
            y_pred = self._model.predict_one(x)
            # Update metric
            self._metric = self._metric.update(y, y_pred)
            # Learn from observation
            self._model = self._model.learn_one(x, y)

            # Store metric history (for plotting)
            continuous_train_mape_history.append(self._metric.get())

            if i % 10000 == 0:
                print("Iter #{}".format(i))
            #print("Iter #{} | Real y: {} | Pred y: {} | metric: {}".format(i, y, y_pred, self._metric))

            # Update iteration counter
            i += 1

        # When there are no more obs the system is either in train or test mode. We need fill the last test/train_region list with the actual iteration
        if adaptative_model_data["operation_mode"] == "train":
            adaptative_model_data["train"]["train_regions"][-1].append(i-1)
        elif adaptative_model_data["test"]["operation_mode"] == "test":
            adaptative_model_data["test"]["test_regions"][-1].append(i-1)

        #print("Training regions: {}".format(adaptative_model_data["train"]["train_regions"]))
        #print("Test regions: {}".format(adaptative_model_data["test"]["test_regions"]))

        # Matplotlib configuration
        mpl.rcParams['figure.figsize'] = (20, 12)
        # Remove top and right frame
        mpl.rcParams['axes.spines.left'] = True
        mpl.rcParams['axes.spines.right'] = False
        mpl.rcParams['axes.spines.top'] = False
        mpl.rcParams['axes.spines.bottom'] = True

        # Conver to np array for being able to substract the list by elements
        continuous_train_mape_history = np.array(continuous_train_mape_history)
        adaptative_model_data["train"]["training_metric_history"] = np.array(adaptative_model_data["train"]["training_metric_history"])
        adaptative_model_data["frozen"]["training_metric_history"] = np.array(adaptative_model_data["frozen"]["training_metric_history"])

        # Create a 2x2 grid of subplots within the same figure
        fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, constrained_layout=False)

        fig.supxlabel('Number of Observations')
        fig.suptitle("{} model - iteration #{}".format(self._type, grid_search_parameters["iteration"]))

        adaptative_mape_difference = adaptative_model_data["frozen"]["training_metric_history"] - adaptative_model_data["train"]["training_metric_history"]

        # Add colored background spans to the plot (train)
        for xmin, xmax in adaptative_model_data["train"]["train_regions"]:
            ax1.axvspan(xmin, xmax, alpha=0.4, color=adaptative_model_data["train"]["train_regions_color"], zorder=0)
        # Add colored background spans to the plot (test)
        for xmin, xmax in adaptative_model_data["test"]["test_regions"]:
            ax1.axvspan(xmin, xmax, alpha=0.4, color=adaptative_model_data["test"]["test_regions_color"], zorder=0)

        # Plot models metrics
        ax1.plot(continuous_train_mape_history, label="continuous_train_mape", color='tab:orange', zorder=1)
        ax1.plot(adaptative_model_data["frozen"]["training_metric_history"], label="frozen_adaptative_training_history", color='tab:blue', zorder=2)
        ax1.plot(adaptative_model_data["train"]["training_metric_history"], label="adaptative_training_history", color='tab:green', zorder=3)
        ax1.set_ylabel("% error", color='k')
        ax1.tick_params(axis='y', labelcolor='k')
        #ax1.set_ylim([-0.5, 350.5])
        #ax1.set_ylim([-0.5, 120.5])
        # Choose the y_lim depending on the models (they have different ranges of errors)
        if self._type == "Time":
            ax1.set_ylim([-0.5, 80.5])
        else:
            ax1.set_ylim([-0.5, 27.5])
        ax1.grid(True)

        # Add a new axis with the diff of the error between the adaptative model and its frozen version
        ax1_diff = ax1.twinx()
        ax1_diff.plot(adaptative_mape_difference, label="adaptative_diff_mape", color='tab:red', zorder=3)
        ax1_diff.set_ylabel("% error", color='tab:red')
        ax1_diff.tick_params(axis='y', labelcolor='tab:red')
        #ax1_diff.set_ylim([-256.5, 85.5])
        #ax1_diff.set_ylim([-80.5, 40.5])
        # Choose the y_lim depending on the models (they have different ranges of errors)
        if self._type == "Time":
            ax1_diff.set_ylim([-120.5, 40.5])
        else:
            ax1_diff.set_ylim([-45.5, 15.5])

        # Adding legend for both y-axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines1_diff, labels1_diff = ax1_diff.get_legend_handles_labels()

        lines1_all = lines1 + lines1_diff
        labels1_all = labels1 + labels1_diff
        ax1.legend(lines1_all, labels1_all, loc="best")

        plt.tight_layout()  # Adjust subplot spacing

        # Save figure
        #python program to check if a path exists
        #if it doesn’t exist we create one
        model_error_figures_dir = "{}/iter_{}".format(grid_search_parameters["output_dir"], grid_search_parameters["iteration"])
        if not os.path.exists(model_error_figures_dir):
            os.makedirs(model_error_figures_dir, exist_ok=True)

        # Save figure
        figure_save_path = "{}/{}.pkl".format(model_error_figures_dir, self._type)
        with open(figure_save_path, 'wb') as f:
            pickle.dump(fig, f)

        # plt.show()

        # Calculate number of observations where the adaptative model is in training stage
        # Convert the list of intervals to a numpy array
        tmp_train_regions = np.array(adaptative_model_data["train"]["train_regions"])

        # Calculate the differences for each interval, including both t0 and t1
        tmp_differences = tmp_train_regions[:, 1] - tmp_train_regions[:, 0] + 1

        # Sum all the differences to get the total difference
        tmp_total_trained_observations = np.sum(tmp_differences)

        return_data = {}
        return_data["continuous"] = {}
        return_data["continuous"]["average_mape"] = np.mean(continuous_train_mape_history)
        return_data["continuous"]["training_intervals"] = i
        return_data["adaptative"] = {}
        return_data["adaptative"]["average_mape"] = np.mean(adaptative_model_data["train"]["training_metric_history"])
        return_data["adaptative"]["training_stages"] = len(adaptative_model_data["train"]["train_regions"])
        return_data["adaptative"]["trained_observations"] = tmp_total_trained_observations
        return_data["adaptative"]["training_regions"] = adaptative_model_data["train"]["train_regions"]
        return_data["frozen"] = {}
        return_data["frozen"]["average_mape"] = np.mean(adaptative_model_data["frozen"]["training_metric_history"])

        return return_data


    def grid_search_train_batch_multiprocessing(self, args):
        """Learn all observations within the 'feature_df' dataframe
           one by one.
        """

        features_df, labels_df, grid_search_parameters = args

        i = 0

        # Adaptative Model Testing
        adaptative_model_data = {}
        adaptative_model_data["model"] = copy.deepcopy(self._model)     # Copia del modelo
        adaptative_model_data["operation_mode"] = "train"       # train or test
        adaptative_model_data["train"] = {}
        adaptative_model_data["train"]["current_iteration"] = 0
        # adaptative_model_data["train"]["nominal_obs_btw_validation"] = 1000             # Numero inicial de observaciones a entrenar antes de comprobar la validation_metric
        # adaptative_model_data["train"]["obs_btw_validation_reduction_factor"] = 0.2  # Factor por el que se reduce el numero de observaciones entre validaciones si en la anterior validación no se supera threshold
        adaptative_model_data["train"]["training_metric"] = river.utils.Rolling(river.metrics.MAPE(), window_size=1000)
        adaptative_model_data["train"]["training_metric_history"] = []
        adaptative_model_data["train"]["validation_metric"] = 100.0              # Metrica de validación empleada para comprobar si el modelo está suficientemente entrenado
        # adaptative_model_data["train"]["validation_threshold"] = 2.5               # Validation_error_threshold. Puntos porcentuales, para MAPE
        adaptative_model_data["train"]["stable_training_error"] = 0              # Contador que indica el numero de veces consecutivas que se supera "validation_threshold"
        # adaptative_model_data["train"]["stable_training_error_threshold"] = 4    # stable_training_error counter threshold. Habrá que ver cuál cuadra
        adaptative_model_data["train"]["train_regions"] = []                     # this list contains tuples indicatingn the training regions of the modes [(t0start, t0end), ..., (tnstart, tnend)]
        adaptative_model_data["train"]["train_regions_color"] = "lightgreen"
        adaptative_model_data["train"]["start_training"] = True  # This flag is used to know if this is the beginning of the train process to store the iteration number on the train_regions variable for posterior plotting
        adaptative_model_data["test"] = {}
        adaptative_model_data["test"]["operation_mode"] = "idle"        # idle or test
        adaptative_model_data["test"]["current_iteration"] = 0
        # adaptative_model_data["test"]["obs_btw_test"] = 1000             # Numero de observaciones a entrenar antes de comprobar la validation_metric
        # adaptative_model_data["test"]["nominal_obs_to_test"] = 200              # Numero inicial de observaciones a entrenar antes de comprobar la validation_metric
        # adaptative_model_data["test"]["obs_to_test_reduction_factor"] = 0.2  # Factor por el que se reduce el numero de observaciones a testear si en el anterior testeo se supera threshold
        adaptative_model_data["test"]["test_metric"] = river.metrics.MAPE()
        adaptative_model_data["test"]["test_metric_history"] = []
        # adaptative_model_data["test"]["test_threshold"] = 2.5               # Validation_error_threshold. Puntos porcentuales, para MAPE
        adaptative_model_data["test"]["significant_error_variation"] = 0              # Contador que indica el numero de veces consecutivas que se supera "validation_threshold"
        # adaptative_model_data["test"]["significant_error_variation_threshold"] = 2    # stable_training_error counter threshold. Habrá que ver cuál cuadraadaptative_model_data["train"]["train_regions"] = []                     # this list contains tuples indicaten the training regions of the modes [(t0start, t0end), ..., (tnstart, tnend)]
        adaptative_model_data["test"]["test_regions"] = []                     # this list contains tuples indicatingn the test in test mode regions of the modes [(t0start, t0end), ..., (tnstart, tnend)]
        adaptative_model_data["test"]["test_regions_color"] = "orange"
        adaptative_model_data["frozen"] = {}
        adaptative_model_data["frozen"]["model"] = copy.deepcopy(self._model)
        adaptative_model_data["frozen"]["training_metric"] = river.utils.Rolling(river.metrics.MAPE(), window_size=1000)
        adaptative_model_data["frozen"]["training_metric_history"] = []
        adaptative_model_data["frozen"]["froze_flag"] = False

        # Get parameters for grid search
        # Train
        adaptative_model_data["train"]["nominal_obs_btw_validation"] = grid_search_parameters["parameters"]["train"]["nominal_obs_btw_validation"]
        adaptative_model_data["train"]["obs_btw_validation_reduction_factor"] = grid_search_parameters["parameters"]["train"]["obs_btw_validation_reduction_factor"]
        adaptative_model_data["train"]["validation_threshold"] = grid_search_parameters["parameters"]["train"]["validation_threshold"]
        adaptative_model_data["train"]["stable_training_error_threshold"] = grid_search_parameters["parameters"]["train"]["stable_training_error_threshold"]
        # Test
        adaptative_model_data["test"]["obs_btw_test"] = grid_search_parameters["parameters"]["test"]["obs_btw_test"]
        adaptative_model_data["test"]["nominal_obs_to_test"] = grid_search_parameters["parameters"]["test"]["nominal_obs_to_test"]
        adaptative_model_data["test"]["test_threshold"] = grid_search_parameters["parameters"]["test"]["test_threshold"]
        adaptative_model_data["test"]["obs_to_test_reduction_factor"] = grid_search_parameters["parameters"]["test"]["obs_to_test_reduction_factor"]
        adaptative_model_data["test"]["significant_error_variation_threshold"] = grid_search_parameters["parameters"]["test"]["significant_error_variation_threshold"]

        # Decide the name of the figure based on the type of the model we are currently working with
        print(self._type)

        # List for storing continuously trained model metrics
        continuous_train_mape_history = []

        for x, y in river.stream.iter_pandas(features_df, labels_df, shuffle=False, seed=42):

            # Cast variables
            # Check if there is cpu_usage data in the observations
            if "user" in x:
                x["user"] = float(x["user"])
                x["kernel"] = float(x["kernel"])
                x["idle"] = float(x["idle"])

            x["Main"] = int(x["Main"])
            x["aes"] = int(x["aes"])
            x["bulk"] = int(x["bulk"])
            x["crs"] = int(x["crs"])
            x["kmp"] = int(x["kmp"])
            x["knn"] = int(x["knn"])
            x["merge"] = int(x["merge"])
            x["nw"] = int(x["nw"])
            x["queue"] = int(x["queue"])
            x["stencil2d"] = int(x["stencil2d"])
            x["stencil3d"] = int(x["stencil3d"])
            x["strided"] = int(x["strided"])
            y = float(y)

            #############################
            ## Test: modelo adaptativo ##
            #############################

            # Predict with the adaptative model
            adaptative_model_y_pred = adaptative_model_data["model"].predict_one(x)
            adaptative_model_data["train"]["training_metric"] = adaptative_model_data["train"]["training_metric"].update(y, adaptative_model_y_pred)
            adaptative_model_data["train"]["training_metric_history"].append(adaptative_model_data["train"]["training_metric"].get())

            # Frozen model
            adaptative_model_y_pred_frozen = adaptative_model_data["frozen"]["model"].predict_one(x)
            adaptative_model_data["frozen"]["training_metric"] = adaptative_model_data["frozen"]["training_metric"].update(y, adaptative_model_y_pred_frozen)
            adaptative_model_data["frozen"]["training_metric_history"].append(adaptative_model_data["frozen"]["training_metric"].get())

            # Only when in training mode
            if adaptative_model_data["operation_mode"] == "train":
                if adaptative_model_data["train"]["start_training"]:
                    # Mark the iteration of the beginning of the training process
                    adaptative_model_data["train"]["train_regions"].append([i])
                    # Set the actual_obs_btw_validation to its nominal value
                    tmp_train_actual_obs_btw_validation = adaptative_model_data["train"]["nominal_obs_btw_validation"]
                    adaptative_model_data["train"]["start_training"] = False

                adaptative_model_data["model"].learn_one(x, y)
                adaptative_model_data["train"]["current_iteration"] += 1


                # Entrenamos el modelo congelado hasta que se congele
                if adaptative_model_data["frozen"]["froze_flag"] is False:
                    adaptative_model_data["frozen"]["model"].learn_one(x, y)

                # Wait until x observations elapse
                if adaptative_model_data["train"]["current_iteration"] == tmp_train_actual_obs_btw_validation:
                    # Check whether the validation_metric is within the training_metric +- validation_threshold
                    if (adaptative_model_data["train"]["training_metric"].get() - adaptative_model_data["train"]["validation_threshold"]) <= adaptative_model_data["train"]["validation_metric"] <= (adaptative_model_data["train"]["training_metric"].get() + adaptative_model_data["train"]["validation_threshold"]):

                        adaptative_model_data["train"]["current_iteration"] = 0
                        # Increment the stable_training_error  counter if the validation_metric is within range
                        adaptative_model_data["train"]["stable_training_error"] += 1
                        tmp_train_actual_obs_btw_validation = round(adaptative_model_data["train"]["nominal_obs_btw_validation"] * (1 - adaptative_model_data["train"]["stable_training_error"] * adaptative_model_data["train"]["obs_btw_validation_reduction_factor"]))
                    else:

                        adaptative_model_data["train"]["current_iteration"] = 0
                        # Clear the stable_training_error counter and freeze the training_metric inside the validation_metric if the validation_metric is outside range
                        adaptative_model_data["train"]["stable_training_error"] = 0
                        tmp_train_actual_obs_btw_validation = adaptative_model_data["train"]["nominal_obs_btw_validation"]
                        adaptative_model_data["train"]["validation_metric"] = adaptative_model_data["train"]["training_metric"].get()

                # Stop training when the stable_training_error counter reaches the threshold indicating the model is well trained
                if adaptative_model_data["train"]["stable_training_error"] == adaptative_model_data["train"]["stable_training_error_threshold"]:
                    adaptative_model_data["operation_mode"] = "test"
                    adaptative_model_data["train"]["start_training"] = True
                    adaptative_model_data["train"]["current_iteration"] = 0
                    adaptative_model_data["train"]["stable_training_error"] = 0
                    tmp_train_actual_obs_btw_validation = adaptative_model_data["train"]["nominal_obs_btw_validation"]
                    # Mark the iteration of the end of the training process
                    adaptative_model_data["train"]["train_regions"][-1].append(i)

                    # Congelamos para siempre
                    adaptative_model_data["frozen"]["froze_flag"] = True

            else:
                if adaptative_model_data["test"]["operation_mode"] == "idle":

                    adaptative_model_data["test"]["current_iteration"] += 1

                    if adaptative_model_data["test"]["current_iteration"] == adaptative_model_data["test"]["obs_btw_test"]:

                        adaptative_model_data["test"]["operation_mode"] = "test"
                        adaptative_model_data["test"]["current_iteration"] = 0
                        adaptative_model_data["test"]["test_metric"] = river.metrics.MAPE()
                        adaptative_model_data["test"]["test_metric_history"] = []
                        adaptative_model_data["test"]["significant_error_variation"] = 0
                        # Mark the iteration of the beginning of the test process
                        adaptative_model_data["test"]["test_regions"].append([i])
                        # Set the actual_obs_to_test to its nominal value
                        tmp_test_actual_obs_to_test = adaptative_model_data["test"]["nominal_obs_to_test"]

                else:

                    adaptative_model_data["test"]["current_iteration"] += 1

                    # Predict with the adaptative model
                    adaptative_model_data["test"]["test_metric"].update(y, adaptative_model_y_pred)
                    adaptative_model_data["test"]["test_metric_history"].append(adaptative_model_data["test"]["test_metric"].get())

                    if adaptative_model_data["test"]["current_iteration"] == tmp_test_actual_obs_to_test:

                        tmp_test_condition = ( (adaptative_model_data["train"]["validation_metric"] - adaptative_model_data["test"]["test_threshold"]) <= adaptative_model_data["test"]["test_metric"].get() <= (adaptative_model_data["train"]["validation_metric"] + adaptative_model_data["test"]["test_threshold"]) )

                        if tmp_test_condition is False:
                            adaptative_model_data["test"]["current_iteration"] = 0
                            adaptative_model_data["test"]["significant_error_variation"] += 1
                            tmp_test_actual_obs_to_test = round(adaptative_model_data["test"]["nominal_obs_to_test"] * (1 - adaptative_model_data["test"]["significant_error_variation"] * adaptative_model_data["test"]["obs_to_test_reduction_factor"]))
                        else:
                            adaptative_model_data["test"]["current_iteration"] = 0
                            adaptative_model_data["test"]["operation_mode"] = "idle"
                            adaptative_model_data["test"]["current_iteration"] = 0
                            tmp_test_actual_obs_to_test = adaptative_model_data["test"]["nominal_obs_to_test"]
                            # Mark the iteration of the end of the test process
                            adaptative_model_data["test"]["test_regions"][-1].append(i)

                    # Go to train when significant_error_variation reaches a threshold indicating the model is not performing properly
                    if adaptative_model_data["test"]["significant_error_variation"] == adaptative_model_data["test"]["significant_error_variation_threshold"]:
                        adaptative_model_data["operation_mode"] = "train"
                        adaptative_model_data["test"]["operation_mode"] = "idle"
                        adaptative_model_data["test"]["current_iteration"] = 0
                        adaptative_model_data["test"]["test_metric"] = river.metrics.MAPE()
                        adaptative_model_data["test"]["test_metric_history"] = []
                        adaptative_model_data["test"]["significant_error_variation"] = 0
                        tmp_test_actual_obs_to_test = adaptative_model_data["test"]["nominal_obs_to_test"]
                        adaptative_model_data["train"]["validation_metric"] = adaptative_model_data["train"]["training_metric"].get()
                        # Mark the iteration of the end of the test process
                        adaptative_model_data["test"]["test_regions"][-1].append(i)

            #print("[Adaptative model - train] train?: {} | cur_iter: {} | train_metric: {} | validation_metric: {} | stable_training_error: {} | stable_training_error_threshold: {}".format(adaptative_model_data["operation_mode"] == "train", adaptative_model_data["train"]["current_iteration"], adaptative_model_data["train"]["training_metric"].get(), adaptative_model_data["train"]["validation_metric"], adaptative_model_data["train"]["stable_training_error"], adaptative_model_data["train"]["stable_training_error_threshold"]))
            #print("[Adaptative model - test]  test?: {} | mode: {} | cur_iter: {} | test_metric: {} | validation_metric: {} | sign_error_variation: {} | sign_error_variation_threshold: {}".format(adaptative_model_data["operation_mode"] == "test", adaptative_model_data["test"]["operation_mode"], adaptative_model_data["test"]["current_iteration"], adaptative_model_data["test"]["test_metric"].get(), adaptative_model_data["train"]["validation_metric"], adaptative_model_data["test"]["significant_error_variation"], adaptative_model_data["test"]["significant_error_variation_threshold"]))

            # Make prediction
            y_pred = self._model.predict_one(x)
            # Update metric
            self._metric = self._metric.update(y, y_pred)
            # Learn from observation
            self._model = self._model.learn_one(x, y)

            # Store metric history (for plotting)
            continuous_train_mape_history.append(self._metric.get())

            if i % 50000 == 0:
                print("[Iteration #{}] {} model -> Iter #{}".format(grid_search_parameters["iteration"], model_type, i))
            #print("Iter #{} | Real y: {} | Pred y: {} | metric: {}".format(i, y, y_pred, self._metric))

            # Update iteration counter
            i += 1

        # When there are no more obs the system is either in train or test mode. We need fill the last test/train_region list with the actual iteration
        if adaptative_model_data["operation_mode"] == "train":
            adaptative_model_data["train"]["train_regions"][-1].append(i-1)
        elif adaptative_model_data["test"]["operation_mode"] == "test":
            adaptative_model_data["test"]["test_regions"][-1].append(i-1)

        #print("Training regions: {}".format(adaptative_model_data["train"]["train_regions"]))
        #print("Test regions: {}".format(adaptative_model_data["test"]["test_regions"]))

        # Matplotlib configuration
        mpl.rcParams['figure.figsize'] = (20, 12)
        # Remove top and right frame
        mpl.rcParams['axes.spines.left'] = True
        mpl.rcParams['axes.spines.right'] = False
        mpl.rcParams['axes.spines.top'] = False
        mpl.rcParams['axes.spines.bottom'] = True

        # Conver to np array for being able to substract the list by elements
        continuous_train_mape_history = np.array(continuous_train_mape_history)
        adaptative_model_data["train"]["training_metric_history"] = np.array(adaptative_model_data["train"]["training_metric_history"])
        adaptative_model_data["frozen"]["training_metric_history"] = np.array(adaptative_model_data["frozen"]["training_metric_history"])

        # Create a 2x2 grid of subplots within the same figure
        fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, constrained_layout=False)

        fig.supxlabel("Number of Observations")
        fig.suptitle("{} model - iteration #{}".format(model_type, grid_search_parameters["iteration"]))

        adaptative_mape_difference = adaptative_model_data["frozen"]["training_metric_history"] - adaptative_model_data["train"]["training_metric_history"]

        # Add colored background spans to the plot (train)
        for xmin, xmax in adaptative_model_data["train"]["train_regions"]:
            ax1.axvspan(xmin, xmax, alpha=0.4, color=adaptative_model_data["train"]["train_regions_color"], zorder=0)
        # Add colored background spans to the plot (test)
        for xmin, xmax in adaptative_model_data["test"]["test_regions"]:
            ax1.axvspan(xmin, xmax, alpha=0.4, color=adaptative_model_data["test"]["test_regions_color"], zorder=0)

        # Plot models metrics
        ax1.plot(continuous_train_mape_history, label="continuous_train_mape", color='tab:orange', zorder=1)
        ax1.plot(adaptative_model_data["frozen"]["training_metric_history"], label="frozen_adaptative_training_history", color='tab:blue', zorder=2)
        ax1.plot(adaptative_model_data["train"]["training_metric_history"], label="adaptative_training_history", color='tab:green', zorder=3)
        ax1.set_ylabel("% error", color='k')
        ax1.tick_params(axis='y', labelcolor='k')
        #ax1.set_ylim([-0.5, 350.5])
        #ax1.set_ylim([-0.5, 120.5])
        # Choose the y_lim depending on the models (they have different ranges of errors)
        if self._type == "Time":
            ax1.set_ylim([-0.5, 80.5])
        else:
            ax1.set_ylim([-0.5, 27.5])
        ax1.grid(True)

        # Add a new axis with the diff of the error between the adaptative model and its frozen version
        ax1_diff = ax1.twinx()
        ax1_diff.plot(adaptative_mape_difference, label="adaptative_diff_mape", color='tab:red', zorder=3)
        ax1_diff.set_ylabel("% error", color='tab:red')
        ax1_diff.tick_params(axis='y', labelcolor='tab:red')
        #ax1_diff.set_ylim([-256.5, 85.5])
        #ax1_diff.set_ylim([-80.5, 40.5])
        # Choose the y_lim depending on the models (they have different ranges of errors)
        if self._type == "Time":
            ax1_diff.set_ylim([-120.5, 40.5])
        else:
            ax1_diff.set_ylim([-45.5, 15.5])

        # Adding legend for both y-axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines1_diff, labels1_diff = ax1_diff.get_legend_handles_labels()

        lines1_all = lines1 + lines1_diff
        labels1_all = labels1 + labels1_diff
        ax1.legend(lines1_all, labels1_all, loc="best")

        plt.tight_layout()  # Adjust subplot spacing

        # Save figure
        #python program to check if a path exists
        #if it doesn’t exist we create one
        model_error_figures_dir = "{}/iter_{}".format(grid_search_parameters["output_dir"], grid_search_parameters["iteration"])
        if not os.path.exists(model_error_figures_dir):
            os.makedirs(model_error_figures_dir, exist_ok=True)

        # Save figure
        figure_save_path = "{}/{}.pkl".format(model_error_figures_dir, self._type)
        with open(figure_save_path, 'wb') as f:
            pickle.dump(fig, f)

        # plt.show()

        # Calculate number of observations where the adaptative model is in training stage
        # Convert the list of intervals to a numpy array
        tmp_train_regions = np.array(adaptative_model_data["train"]["train_regions"])

        # Calculate the differences for each interval, including both t0 and t1
        tmp_differences = tmp_train_regions[:, 1] - tmp_train_regions[:, 0] + 1

        # Sum all the differences to get the total difference
        tmp_total_trained_observations = np.sum(tmp_differences)

        return_data = {}
        return_data["continuous"] = {}
        return_data["continuous"]["average_mape"] = np.mean(continuous_train_mape_history)
        return_data["continuous"]["training_intervals"] = i
        return_data["adaptative"] = {}
        return_data["adaptative"]["average_mape"] = np.mean(adaptative_model_data["train"]["training_metric_history"])
        return_data["adaptative"]["training_stages"] = len(adaptative_model_data["train"]["train_regions"])
        return_data["adaptative"]["trained_observations"] = tmp_total_trained_observations
        return_data["adaptative"]["training_regions"] = adaptative_model_data["train"]["train_regions"]
        return_data["frozen"] = {}
        return_data["frozen"]["average_mape"] = np.mean(adaptative_model_data["frozen"]["training_metric_history"])

        return return_data

    def train_single(self, features, label, i):
        """Learn just one observation from a dictionary.
        """
        # Learn one observation
        self._model.learn_one(features, label)

    def predict_one(self, features_dict):
        """Make a one prediction from features (received as a dictionary)."""

        # Make prediction
        y_pred = self._model.predict_one(features_dict)

        return y_pred

    def update_metric(self, real_value, predicted_value):
        """Updates the error metric. It is done in-place!"""

        self._metric = self.metric.update(real_value, predicted_value)

        return self._metric
    
    def test_batch(self, features_df, labels_df, metric=None):
        """Test the model on a set of observations reveived as a dataframe."""

        metric = river.metrics.MAPE() if metric is None else metric

        print(metric)

        # print("\nPredict One Time: {} ms (mean) | {} ms (last)".format(
        #     (predict_one_total_time/1000000)/len(features_df),
        #     (t1_predict_one-t0_predict_one)/1000000)
        # )
        # print("Metrics Update Time: {} ms (mean) | {} ms (last)\n".format(
        #     (update_metric_total_time/1000000)/len(features_df),
        #     (t1_update_metric-t0_update_metric)/1000000)
        # )

        # Poster annual meeting 15

        i = 0
        am_15_y_pred = []
        am_15_y = []

        metric_history = []

        for x, y in river.stream.iter_pandas(features_df, labels_df, shuffle=False, seed=42):

            x["user"] = float(x["user"])
            x["kernel"] = float(x["kernel"])
            x["idle"] = float(x["idle"])
            x["Main"] = int(x["Main"])
            x["aes"] = int(x["aes"])
            x["bulk"] = int(x["bulk"])
            x["crs"] = int(x["crs"])
            x["kmp"] = int(x["kmp"])
            x["knn"] = int(x["knn"])
            x["merge"] = int(x["merge"])
            x["nw"] = int(x["nw"])
            x["queue"] = int(x["queue"])
            x["stencil2d"] = int(x["stencil2d"])
            x["stencil3d"] = int(x["stencil3d"])
            x["strided"] = int(x["strided"])
            y = float(y)

            # Make a prediction
            y_pred = self._model.predict_one(x)

            # Update metric
            metric = metric.update(y, y_pred)

            # Store metric history (for plotting)
            metric_history.append(metric.get())

            # predict_one_total_time += t1_predict_one - t0_predict_one
            # update_metric_total_time += t1_update_metric - t0_update_metric
            # learn_one_total_time += t1_learn_one - t0_learn_one

            i += 1

            print("Iter #{} | Real y: {} | Pred y: {} | metric: {}".format(i, y, y_pred, metric))

            # Test AM_15
            am_15_y_pred.append(y_pred)
            am_15_y.append(y)

        # AM15 test
        color1 = "#D4CC47"
        color2 = "#7C4D8B"

        def hex_to_RGB(hex_str):
            """ #FFFFFF -> [255,255,255]"""
            #Pass 16 to the integer function for change of base
            return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]
        def get_color_gradient(c1, c2, n):
            """
            Given two hex colors, returns a color gradient
            with n colors.
            """
            assert n > 1
            c1_rgb = np.array(hex_to_RGB(c1))/255
            c2_rgb = np.array(hex_to_RGB(c2))/255
            mix_pcts = [x/(n-1) for x in range(n)]
            rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
            return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]


        mpl.rcParams['figure.figsize'] = (20, 12)

        # Remove top and right frame
        mpl.rcParams['axes.spines.left'] = True
        mpl.rcParams['axes.spines.right'] = False
        mpl.rcParams['axes.spines.top'] = False
        mpl.rcParams['axes.spines.bottom'] = True

        am_x_values = np.linspace(0,8,150)
        #plt.scatter(am_15_y, am_15_y_pred, s=15, c=blue_color_matlab, marker=".")
        first_plot_start_index = 0
        first_plot_end_index = int(len(am_15_y_pred)/10)
        second_plot_start_index = int(3*len(am_15_y_pred)/10)
        second_plot_end_index = int(4*len(am_15_y_pred)/10)
        third_plot_start_index = int(19*len(am_15_y_pred)/20)
        third_plot_end_index = len(am_15_y_pred)

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(1, 1, 1)
        #plt.scatter(am_15_y[first_plot_start_index:first_plot_end_index], am_15_y_pred[first_plot_start_index:first_plot_end_index], s=15, marker=".", c=blue_color_matlab)
        plt.scatter(am_15_y[first_plot_start_index:first_plot_end_index], am_15_y_pred[first_plot_start_index:first_plot_end_index], s=15, marker=".", color=get_color_gradient(color1, color2, len(am_15_y_pred[first_plot_start_index:first_plot_end_index])))
        plt.scatter(am_x_values, am_x_values, s=2, c='black', marker=".")

        # Major ticks every 20, minor ticks every 5
        major_ticks = np.arange(0, 9, 2)
        minor_ticks = np.arange(0, 9, 0.5)

        ax1.set_xticks(major_ticks)
        ax1.set_xticks(minor_ticks, minor=True)
        ax1.set_yticks(major_ticks)
        ax1.set_yticks(minor_ticks, minor=True)

        # And a corresponding grid
        ax1.grid(which='both')

        # Or if you want different settings for the grids:
        ax1.grid(which='minor', alpha=0.5, linestyle=':')
        ax1.grid(which='major', alpha=0.7, linestyle='-')

        plt.ylim(0, 15.01)
        plt.xlim(0, 15.01)

        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)

        #ax1.set_title("Performance Model (RT)", fontsize=26, fontweight="bold")
        ax1.set_ylabel("Predicted Execution Time (ms)", fontsize=36)
        ax1.set_xlabel("Actual Execution Time (ms)", fontsize=36)

        plt.show()
        #fig1.savefig("./AM15/fig1.svg", format = 'svg', dpi=600)

        # ax1.text(6.5,1, 'RMSE = {:.4f}\nMAPE = {:.3f}'.format(final_rmse, final_mape), fontsize = 22,
        #    bbox = dict(facecolor = 'white', alpha = 1))

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1, 1, 1)
        #plt.scatter(am_15_y[second_plot_start_index:second_plot_end_index], am_15_y_pred[second_plot_start_index:second_plot_end_index], s=15, marker=".", c=blue_color_matlab)
        plt.scatter(am_15_y[second_plot_start_index:second_plot_end_index], am_15_y_pred[second_plot_start_index:second_plot_end_index], s=15, marker=".", color=get_color_gradient(color1, color2, len(am_15_y_pred[second_plot_start_index:second_plot_end_index])))
        plt.scatter(am_x_values, am_x_values, s=2, c='black', marker=".")

        # Major ticks every 20, minor ticks every 5
        major_ticks = np.arange(0, 9, 2)
        minor_ticks = np.arange(0, 9, 0.5)

        ax2.set_xticks(major_ticks)
        ax2.set_xticks(minor_ticks, minor=True)
        ax2.set_yticks(major_ticks)
        ax2.set_yticks(minor_ticks, minor=True)

        # And a corresponding grid
        ax2.grid(which='both')

        # Or if you want different settings for the grids:
        ax2.grid(which='minor', alpha=0.5, linestyle=':')
        ax2.grid(which='major', alpha=0.7, linestyle='-')

        plt.ylim(0, 15.01)
        plt.xlim(0, 15.01)

        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)

        #ax2.set_title("Performance Model (RT)", fontsize=26, fontweight="bold")
        ax2.set_ylabel("Predicted Execution Time (ms)", fontsize=36)
        ax2.set_xlabel("Actual Execution Time (ms)", fontsize=36)

        # ax2.text(6.5,1, 'RMSE = {:.4f}\nMAPE = {:.3f}'.format(final_rmse, final_mape), fontsize = 22,
        #    bbox = dict(facecolor = 'white', alpha = 1))

        plt.show()
        #fig2.savefig("./AM15/fig2.svg", format = 'svg', dpi=600)

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(1, 1, 1)
        #plt.scatter(am_15_y[third_plot_start_index:third_plot_end_index], am_15_y_pred[third_plot_start_index:third_plot_end_index], s=15, marker=".", c=blue_color_matlab)
        plt.scatter(am_15_y[third_plot_start_index:third_plot_end_index], am_15_y_pred[third_plot_start_index:third_plot_end_index], s=15, marker=".", color=get_color_gradient(color1, color2, len(am_15_y_pred[third_plot_start_index:third_plot_end_index])))
        plt.scatter(am_x_values, am_x_values, s=2, c='black', marker=".")

        # Major ticks every 20, minor ticks every 5
        major_ticks = np.arange(0, 9, 2)
        minor_ticks = np.arange(0, 9, 0.5)

        ax3.set_xticks(major_ticks)
        ax3.set_xticks(minor_ticks, minor=True)
        ax3.set_yticks(major_ticks)
        ax3.set_yticks(minor_ticks, minor=True)

        # And a corresponding grid
        ax3.grid(which='both')

        # Or if you want different settings for the grids:
        ax3.grid(which='minor', alpha=0.5, linestyle=':')
        ax3.grid(which='major', alpha=0.7, linestyle='-')

        plt.ylim(0, 15.01)
        plt.xlim(0, 15.01)

        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)

        #ax3.set_title("Performance Model (RT)", fontsize=26, fontweight="bold")
        ax3.set_ylabel("Predicted Execution Time (ms)", fontsize=36)
        ax3.set_xlabel("Actual Execution Time (ms)", fontsize=36)

        # ax2.text(6.5,1, 'RMSE = {:.4f}\nMAPE = {:.3f}'.format(final_rmse, final_mape), fontsize = 22,
        #    bbox = dict(facecolor = 'white', alpha = 1))

        plt.show()

        test_fig = plt.figure()
        plt.plot(metric_history)
        plt.show()

        return metric

    @property
    def metric(self):
        """Return the training metric of the model so far."""
        return self._metric

    def get_description(self):
        """(TEST) Print a model identifier."""
        print("This is the {} Model".format(self._type))


class PowerModel(Model):
    """Power model as a Singleton class."""

    def __init__(self, power_type, input_model=None):

        # Define the model and metrics used
        if input_model is None:
            tmp_model = (
                river.preprocessing.StandardScaler() |
                river.tree.HoeffdingAdaptiveTreeRegressor(
                    max_depth=100,
                    grace_period=50,
                    model_selector_decay=0.05,
                    seed=42
                )
            )
        else:
            tmp_model = input_model

        tmp_metric = river.utils.Rolling(river.metrics.MAPE(), window_size=1000)

        if power_type not in ["PS", "PL", "PS+PL"]:
            raise ValueError(f"Invalid type of Power model: {power_type}. Must be one of: PS, PL, PS+PL")
        tmp_type = power_type

        super().__init__(tmp_model, tmp_metric, tmp_type)


class TimeModel(Model):
    """Performance model as a Singleton class."""

    def __init__(self, input_model=None):

        # Define the model and metrics used
        if input_model is None:
            tmp_model = river.forest.ARFRegressor(seed=42, max_features=None, grace_period=50, n_models = 5, max_depth=100, model_selector_decay=0.05)
        else:
            tmp_model = input_model

        tmp_metric = river.utils.Rolling(river.metrics.MAPE(), window_size=1000)

        tmp_type = "Time"

        super().__init__(tmp_model, tmp_metric, tmp_type)


if __name__ == "__main__":

    # Instantiate each model
    ps_power_model = PowerModel("PS")
    pl_power_model = PowerModel("PL")
    time_model = TimeModel()

    # Print the identifier of each model
    ps_power_model.get_description()
    pl_power_model.get_description()
    time_model.get_description()
