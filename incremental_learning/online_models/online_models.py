#!/usr/bin/env python3

"""
Online Modeling Management

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : October 2022
Description : This class handle each online model (top power, bottom power,
              and time) as a whole for the initialization, train and test
              stages.

"""

import os
import json
import shutil
import itertools
import multiprocessing
import pickle
import copy
import numpy as np
import pandas as pd
import river
# TODO: delete
import time

# for working as a module
from . import models
from . import processing

# for internal testing
# import models
# import processing


# Create a encoder to properly encode numpy data types on JSON format
class MyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(MyJSONEncoder, self).default(obj)


class OnlineModels():
    """Class that groups and handle each model as a whole."""

    def __init__(self,
                 power_rails,
                 lock,
                 input_models=None,
                 train_mode="adaptive",
                 capture_all_traces=False):
        """
        Initialize the OnlineModels class.

        Parameters
        ----------
        power_rails : str
            The power_rails to be used. It can be "mono" or "dual".
        lock : multiprocessing.Lock
            Lock to be used for thread safety.
        input_models : list, optional
            List of input models to be used. The default is None.
        train_mode : str
            The training mode to be used for the online learning process ("adaptive", "always_train", "one_train").
        capture_all_traces : bool
            Flag to indicate if all the traces are going to be captured, not just the ones that are going to be used for training.
        """

        # Check if the power_rails is valid and set the number of models
        num_models = 3 if power_rails == "dual" else 2 if power_rails == "mono" else None
        if num_models is None:
            raise ValueError(f"power_rails {power_rails} not supported.")
        self._power_rails = power_rails

        # Get the input models
        if input_models is None:
            input_models = [None] * num_models
        elif len(input_models) != num_models:
            raise ValueError("The number of input models does not match the number of models.")

        # Initialize the models
        if power_rails == "dual":
            self._models = [
                models.PowerModel("PS",input_model=input_models[0], train_mode=train_mode, capture_all_traces=capture_all_traces),
                models.PowerModel("PL",input_model=input_models[1], train_mode=train_mode, capture_all_traces=capture_all_traces),
                models.TimeModel(input_model=input_models[2], train_mode=train_mode, capture_all_traces=capture_all_traces)
                ]
        elif power_rails == "mono":
            self._models = [
                models.PowerModel("PS+PL",input_model=input_models[0], train_mode=train_mode, capture_all_traces=capture_all_traces),
                models.TimeModel(input_model=input_models[1], train_mode=train_mode, capture_all_traces=capture_all_traces)
                ]
        else:
            raise ValueError(f"power_rails {power_rails} not supported.")

        # Set the training mode
        self._train_mode = train_mode
        self._capture_all_traces = capture_all_traces

        # Set the methods
        self.update_models = self._update_models
        self.predict_one = self._predict_one
        self.get_metrics = self._get_metrics
        self.test = self._test
        self.print_training_monitor_info = self._print_training_monitor_info
        self.grid_search_train = self._grid_search_adaptative_train_zcu
        self.grid_search_train_multiprocessing = self._grid_search_adaptative_train_zcu_multiprocessing

        # Particular method for capturing all traces
        if self._capture_all_traces is True:
            self.idle_update_metrics = self._idle_update_metrics

        self._lock = lock

    def _preprocess_dataframe(self, df, power_rails):
        """Preprocess a dataframe for train/test."""

        # Format the dataframe ([features_df, x_model_labels_df, y_model_labels_df, ...])
        formated_dataset = processing.dataset_formating(df, power_rails)

        # Concatenate the DataFrames along the columns axis
        concatenated_labels_df = pd.concat(formated_dataset[1:], axis=1)

        return formated_dataset[0], concatenated_labels_df

    def _features_labels_accommodation(self, features, labels):
        """Perform accomodation on features and labels. Type casting..."""
        # Cast variables
        # Check if there is cpu_usage data in the observations
        if "user" in features:
            features["user"] = float(features["user"])
            features["kernel"] = float(features["kernel"])
            features["idle"] = float(features["idle"])

        features["Main"] = int(features["Main"])
        features["aes"] = int(features["aes"])
        features["bulk"] = int(features["bulk"])
        features["crs"] = int(features["crs"])
        features["kmp"] = int(features["kmp"])
        features["knn"] = int(features["knn"])
        features["merge"] = int(features["merge"])
        features["nw"] = int(features["nw"])
        features["queue"] = int(features["queue"])
        features["stencil2d"] = int(features["stencil2d"])
        features["stencil3d"] = int(features["stencil3d"])
        features["strided"] = int(features["strided"])

        # Get each model label
        labels = [float(labels[key]) for key in labels]

        return features, labels

    def _get_models_state(self):
        """Get the state of the Training Monitor of all the models. [mode, changed]"""
        return [value for model in self._models for value in model.get_state()]

    def _get_models_mode(self):
        """Get the operation modes of the Training Monitor of all the models."""

        # Get the models mode (even indexes)
        return self._get_models_state()[::2]

    def _sync_models(self):
        """Keep the models in sync"""

        # Read operation mode and if the stage has changed
        models_state = self._get_models_state()
        models_mode = models_state[::2]
        models_changed = models_state[1::2]

        # When no stage changed signal just return the model modes
        # (meaning no models diverge from the "normal" path)
        if True not in models_changed:
            return models_mode

        # Get model types for reference
        models_types = [model._type for model in self._models]

        # Create a data structure containing model information to sync them {model: (mode, changed)}
        sync_info = {
            key: (mode, changed)
            for key, mode, changed in zip(models_types, models_mode, models_changed)
            }

        # This dictionary is used for sorting the models based on their mode
        # It is useful for processing them later
        # Since "idle" models cannot affect "test" which cannot affect "train" models
        # So if you go in order you do not have to care about previous models affecting next ones
        hierarchy = {"idle": 0, "test": 1, "train": 2}

        # Sort the dictionary
        sync_info = dict(sorted(sync_info.items(), key=lambda pair: hierarchy[pair[1][0]]))

        # Data structure that will contain the actions to perform in the sync
        stage_to_sync = "empty"
        models_to_sync = ["top", "bottom", "time"]
        models_to_clear = None

        # This algorithm synchronized the models so they are always on the same operation mode
        #
        # It is based on the actual modes of the models as well as a stage_changed flag that
        # indicates if the model has just changed to that mode.
        #
        # This is important because, if we want the models to be in sync, a change in the mode of
        # a model must trigger a change in the others.
        #
        # However, not every mode change trigger changes on other models.
        # There are priorities among the operation modes (train > test > idle).
        #
        # The rules are the following, in order of priority:
        #
        # 1 When a model goes to "train" (or when it was on train but restarts the training process)
        #   it switches all the other models to "train".
        # 2 When a model goes to "idle" (remember here there are no models in "train" due to (1)):
        #     - If any model is in "test" nothing happens, since it has to wait for the "test" model
        #       to finish testing.
        #     - But if the other models are already in "idle" (because they were waiting for him to
        #       finish testing). It makes them restart the "idle" phase. So they keep in sync.
        #
        # The models have been sorted from idle to train because is more convenient to proccess
        # higher priority models last, since they overwrite any changes done by lower-priority ones
        #
        #
        # Approach:
        #
        # stage_to_sync: Is a variable that will indicate to which stage the models have to sync to
        # models_to_sync: It indicates which models have to be synced to stage_to_sync.
        #
        # - The alg iterate over each sorted model.
        # - The model only actuate when their stage_changed flag is True. Otherwise they depend on
        #   the other models.
        # - If its new mode of operation is the same as the stage_to_sync it removes himself from
        #   models_to_sync. Since its already on that mode.
        # - If its new mode of operation is not the same as stage_to_sync it overwrites it with its
        #   operation mode, if and only if its mode of operation has higher hierarchy that the one
        #   in stage_to_sync (e.g., "train" will change "test" or "idle", but not viceversa)
        #
        # * The priority thing is achived with the sorting. Latest models (higher priority) will
        #   overwrite changes made by previous. That why the code does not check priorities.
        #
        # * There are particular caviats that can be seen in the actual algorithm
        #
        for model, (mode, changed) in sync_info.items():
            if mode == "idle":
                if changed:
                    # When the mode is "idle" and has just changed
                    # - In case stage_to_sync is "idle". Just removes himself from models_to_sync
                    # - In case stage_to_sync is any other. Set its mode as stage_to_sync and place
                    #   The other models in the models_to_sync list
                    if stage_to_sync == "idle":
                        models_to_sync.remove(model)
                    else:
                        stage_to_sync = mode
                        models_to_sync = [x for x in models_types if x != model]
                else:
                    # Do nothing
                    pass
            elif mode == "test":
                if changed:
                    # This cannot happen. Since that flag is not set when changing to test.
                    # Because that change sould not trigger changes in other models.
                    print(
                        f"[Sync] mode: {mode} "
                        f"and changed: {changed} "
                        f"Cannot happend (model: {model})"
                    )
                    exit()
                else:
                    if stage_to_sync == "idle":
                        # When the mode is "test"
                        # - In case stage_to_sync is "idle".
                        #   (Scenario that happens when all the models where in "test" but some go
                        #   to "idle" while at least one remains in "test")
                        #
                        # What it happens is that the models that are in "idle" as a new stage,
                        # (which are the models that have change stage_to_chage to "idle"),are
                        # cleared. Which means changing their stage_changed flag to Flase. Since
                        # in the particular case of model change to idle while other still in "test"
                        # the models in "test" should not be disturbed.
                        #
                        # To find the models that are in idle. We just check which models are not
                        # in the models_to_sync list, since models in "idle" have remove themshelf
                        # from that list earlier.
                        # Those models are placed in the models_to_clear list
                        models_to_clear = [x for x in models_types if x not in models_to_sync]
                        stage_to_sync = "empty"
                        models_to_sync = models_types.copy()
                    else:
                        # Do nothing
                        pass
            elif mode == "train":
                if changed:
                    # When the mode is "train" and has just changed
                    # - In case stage_to_sync is "train". Just removes himself from models_to_sync
                    # - In case stage_to_sync is any other. Set its mode as stage_to_sync and place
                    #   The other models in the models_to_sync list
                    if stage_to_sync == "train":
                        models_to_sync.remove(model)
                    else:
                        stage_to_sync = mode
                        models_to_sync = [x for x in models_types if x != model]
                else:
                    # Do nothing
                    pass
            else:
                print(f"[Sync] mode: {mode} - Error")
                exit()

        print("final stage_to_sync:", stage_to_sync)
        print("final models_to_sync:", models_to_sync)
        print("final models_to_clear:", models_to_clear)

        # Check if there are any models to be cleared
        if models_to_clear is not None:
            # Been here means some models went to idle by at least one is still in test
            # We just clear the stage_changed variable when set
            for model in models_to_clear:
                self._models[models_types.index(model)].clear_stage_changed_flag()
            # Return the models modes
            return models_mode

        # Make the syncing
        # Models marked as "to_sync" are reset to the proper stage
        # Models not marked as "to_sync" have their stage_changed flag cleared
        for model in self._models:
            if model._type in models_to_sync:
                model.reset_to_stage(stage_to_sync)
            else:
                model.clear_stage_changed_flag()

        # Return models modes
        return self._get_models_mode()

    # TODO: Remove iteration (for plotting)
    def _idle_update_metrics(self, train_df):
        """Update the Training Monitor when in idle (for the paper)."""

        # Preprocess dataframe
        features_df, concatenated_labels_df = self._preprocess_dataframe(train_df, self._power_rails)

        # Acquire lock - Freeze models
        # TODO: el deepcopy cada vez tarda más... segundos
        # TODO: Tener en cuenta que podría pasar algo cuando se haga un predict_one mientras se entrena
        # TODO: Aunque en realidad... sería solo leer. Hay que mirar implementación en riverml
        # with self._lock:
        #     self._top_power_prediction_model = copy.deepcopy(self._top_power_model)
        #     self._bottom_power_prediction_model = copy.deepcopy(self._bottom_power_model)
        #     self._time_prediction_model = copy.deepcopy(self._time_model)

        # Loop over the observations
        for features, labels in river.stream.iter_pandas(features_df, concatenated_labels_df, shuffle=False, seed=42):

            # Features and labels accommodation
            features, labels = self._features_labels_accommodation(features, labels)

            for model, label in zip(self._models, labels):
                # Make a prediction
                y_pred = model.predict_one(features)
                # Update metric
                model.update_metric(label, y_pred)
                # Update the Training Metric
                model._training_monitor.train_training_metric.update(label, y_pred)
                model._training_monitor.train_training_metric_history.append(model._training_monitor.train_training_metric.get())

    # TODO: Remove iteration (for plotting)
    def _train_models(self, train_df, iteration):
        """(Thread Safe)Test the models with a dataframe. Update the Training Monitor."""

        # Preprocess dataframe
        features_df, concatenated_labels_df = self._preprocess_dataframe(train_df, self._power_rails)

        # Get the number of observations in the df
        obs_to_train = len(train_df)

        # Acquire lock - Freeze models
        # TODO: el deepcopy cada vez tarda más... segundos
        # TODO: Tener en cuenta que podría pasar algo cuando se haga un predict_one mientras se entrena
        # TODO: Aunque en realidad... sería solo leer. Hay que mirar implementación en riverml
        # with self._lock:
        #     self._top_power_prediction_model = copy.deepcopy(self._top_power_model)
        #     self._bottom_power_prediction_model = copy.deepcopy(self._bottom_power_model)
        #     self._time_prediction_model = copy.deepcopy(self._time_model)

        # Loop over the observations
        count = 0
        for count, (features, labels) in enumerate(
            river.stream.iter_pandas(features_df, concatenated_labels_df, shuffle=False, seed=42),
            start=1):

            # Features and labels accommodation
            features, labels = self._features_labels_accommodation(features, labels)

            for model, label in zip(self._models, labels):
                # Make a prediction
                y_pred = model.predict_one(features)
                # Special case when in "one_train" mode, skip training after first training phase
                if model._training_monitor.stop_training_forever is False:
                    # Train the model
                    model.train_single(features, label, iteration)
                    # Update metric
                    model.update_metric(label, y_pred)
                # Update the Training Monitor
                model.update_state(label, y_pred, iteration)

            # Synchronize all the models
            models_mode = self._sync_models()

            # Decide the next mode
            if len(set(models_mode)) > 1:
                # Models cannot be in different models when in training of after training phase
                print("[Train] Error, not every model goes to train...")
                exit()
            else:
                # Since all the models are in the same mode. We get just the first one
                next_mode = models_mode[0]

            # Increase the iteration counter
            iteration += 1

            # Stop training when no longer in "train" mode
            if next_mode != "train":
                # Stop training
                break

        # Acquire lock - Release models
        # with self._lock:
        #     self._top_power_prediction_model = self._top_power_model
        #     self._bottom_power_prediction_model = self._bottom_power_model
        #     self._time_prediction_model = self._time_model

        # Return the next operation mode, de amount of not trained obs, and the iteration count
        return iteration, next_mode, obs_to_train - count

    # TODO: Remove iteration (for plotting)
    def _test_models(self, test_df, iteration):
        """(Thread Safe) Test the models on a dataframe. Update the Training Monitor."""

        # Preprocess dataframe
        features_df, concatenated_labels_df = self._preprocess_dataframe(test_df, self._power_rails)

        # Get the number of observations in the df
        obs_to_test = len(test_df)

        # Some models could be already in idle state, need to be taken into account
        models_mode = self._get_models_mode()

        # Acquire lock - Freeze models
        # TODO: el deepcopy cada vez tarda más... segundos
        # TODO: Tener en cuenta que podría pasar algo cuando se haga un predict_one mientras se entrena
        # TODO: Aunque en realidad... sería solo leer. Hay que mirar implementación en riverml
        # with self._lock:
        #     self._top_power_prediction_model = copy.deepcopy(self._top_power_model)
        #     self._bottom_power_prediction_model = copy.deepcopy(self._bottom_power_model)
        #     self._time_prediction_model = copy.deepcopy(self._time_model)

        # Loop over the observations. Using enumerate to count while going through
        count = 0
        for count, (features, labels) in enumerate(
            river.stream.iter_pandas(features_df, concatenated_labels_df, shuffle=False, seed=42),
            start=1):

            # Features and labels accommodation
            features, labels = self._features_labels_accommodation(features, labels)

            for model, label, mode in zip(self._models, labels, models_mode):
                if mode ==  "test":
                    # Test when in "test" mode
                    y_pred = model.predict_one(features)
                    model.update_metric(label, y_pred)
                    model.update_state(label, y_pred, iteration)
                elif mode == "idle":
                    if self._capture_all_traces is True:
                        y_pred = model.predict_one(features)
                        model.update_metric(label, y_pred)
                        # Update training metric
                        model._training_monitor.train_training_metric.update(label, y_pred)
                        model._training_monitor.train_training_metric_history.append(model._training_monitor.train_training_metric.get())
                    # Increment "idle" current iteration when "idle"
                    model.increment_idle_phase()

            # Synchronize all the models
            models_mode = self._sync_models()

            # Define next mode for the models
            if len(set(models_mode)) > 1:
                # When not all are in the same mode.
                # if al least one model is in "test" we define next mode as "test"
                if "test" in models_mode:
                    next_mode = "test"
            else:
                # When all models have the same mode we just get the first one
                next_mode = models_mode[0]

            # Update iteration
            iteration += 1

            # End testing phasd whenever the operation mode is no longer "test"
            if next_mode != "test":
                # Stop testing
                break

        # Acquire lock - Release models
        # with self._lock:
        #     self._top_power_prediction_model = self._top_power_model
        #     self._bottom_power_prediction_model = self._bottom_power_model
        #     self._time_prediction_model = self._time_model

        # Return the next operation mode, de amount of not tested obs, and the iteration count
        return iteration, next_mode, obs_to_test - count

    def _predict_one(self, features_dict):
            """Make a prediction based on given features for each model."""

            # Make a prediction for each model
            return [model.predict_one(features_dict) for model in self._models]

    def _test(self, test_df, metrics=None):
        """Format the input observation dataframe and test each model on each
           of the observations.
        """

        # Define the metrics to be used
        if metrics is None:
            metrics = [None] * len(self._models)

        # Preprocess the dataframe
        formated_dataset = processing.dataset_formating(test_df, self._power_rails)

        # Get the features and labels
        features_df = formated_dataset[0]
        labels_df_list = formated_dataset[1:]

        # Test each model on the observations
        return [model.test_batch(features_df, labels_df, metric) for model, labels_df, metric in zip(self._models, labels_df_list, metrics)]

    def _get_metrics(self):
        """Returns the training error metrics for each model so far."""
        return [model.metric for model in self._models]

    # TODO lock para predicciones y entrenamiento?
    def _update_models(self, observations_df, iteration):
        """Update the models. Deciding whether to train or test."""

        # Get models info used to decide whether to train
        models_mode = self._get_models_mode()

        # Decide whether to train or test
        if all(mode == "train" for mode in models_mode):
            # Train models
            iteration, mode, non_processed_obs = self._train_models(observations_df,iteration)
        elif "test" in models_mode:
            # Test models
            iteration, mode, non_processed_obs = self._test_models(observations_df,iteration)
        else:
            raise ValueError("All models are in idle mode. This should not happen.")

        # print("[Update Models] Observaciones desperdiciadas:", non_processed_obs)

        # Get next operation training monitor info to signal indications to the setup side
        models_mode = self._get_models_mode()

        # When mode is "test" it could mean some models are idle an at least one in test
        # That case need to be identify to know which models is in testing and get its information
        # to tell the setup side the number of observations it has to gather next

        # Variable that will store the index of the model different to the others
        # (top: 0, bottom: 1, time: 2)
        # Set to 0 because when this index doesnt change it means all the models are in the same
        # state, therefore all have the same info, so we get the info from the first one
        model_index = 0
        # Get the Training Monitor info from the selected model
        training_monitor_info = self._models[model_index].get_info()
        if mode == "test":
            # Get the number of different models within the models
            if len(set(models_mode)) > 1:
                # When there is more than one mode it means some models are in "idle" while others
                # are in "test"
                # We need to find the one in "test"
                model_index = models_mode.index("test")
            # Get the Training Monitor info from the selected model
            training_monitor_info = self._models[model_index].get_info()
        # We are moving from train or test to an idle stage
        elif mode == "idle":

            # Since we are moving to a new fresh idle stage the idle obs left should be
            # way more than the non-processed ones
            if training_monitor_info["minimum_test_idle_obs_left"] <= non_processed_obs:
                print(
                    """[Update Models] More non-processed obs than obs to be idle.")
                    This shouldn't happen since been here means going from train/test to idle now.
                    So we have the whole testing phase ahead of us"""
                    )
                exit()

            # - Since we are in the idle stage we have to return info about how
            #   many obs the setup should go without measuring and reporting
            # - We are going also to update the Training Monitor asuming this
            #   following idle phase has been carried properly.
            #   So when the setup reports back after the idle phase (for test)
            #   we have the Training Monitor set for the test phase

            # Calculate the number of observations the setup has to wait in the idle phase
            num_obs_to_be_idle = \
                training_monitor_info["minimum_test_idle_obs_left"] - non_processed_obs
            # Update the iteration value with the minimum required to end the test-idle part
            iteration += training_monitor_info["minimum_test_idle_obs_left"]
            # Signal the end of the idle phase (we are assuming the setup works properly)
            for model in self._models:
                model.end_idle_phase(iteration)

            # Return action to be commanded to the setup
            return iteration, mode, num_obs_to_be_idle

        # When in testing or training mode the setup has to keep generating traces
        if mode == "train":
            return iteration, mode, training_monitor_info["minimum_train_obs_left"]
        elif mode == "test":
            return iteration, mode, training_monitor_info["minimum_test_test_round_obs_left"]

    def _print_training_monitor_info(self, output_path):


        # TODO: Not used on-chip. Handle this properly
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        # Matplotlib configuration
        mpl.rcParams['figure.figsize'] = (20, 12)
        # Remove top and right frame
        mpl.rcParams['axes.spines.left'] = True
        mpl.rcParams['axes.spines.right'] = False
        mpl.rcParams['axes.spines.top'] = False
        mpl.rcParams['axes.spines.bottom'] = True

        for model in self._models:
            # Create a 2x2 grid of subplots within the same figure
            fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, constrained_layout=False)

            fig.supxlabel('Number of Observations')
            fig.suptitle('Error Metrics!!')

            # Add colored background spans to the plot (train)
            for xmin, xmax in model._training_monitor.train_train_regions:
                ax1.axvspan(
                    xmin, xmax, alpha=0.4,
                    color=model._training_monitor.train_train_regions_color,
                    zorder=0
                    )
            # Add colored background spans to the plot (test)
            for xmin, xmax in model._training_monitor.test_test_regions:
                ax1.axvspan(
                    xmin, xmax, alpha=0.4,
                    color=model._training_monitor.test_test_regions_color,
                    zorder=0
                    )
            # Plot model metrics
            ax1.plot(
                model._training_monitor.train_training_metric_history,
                label="adaptative_training_history",
                color='tab:green',
                zorder=2
                )
            if isinstance(model, models.PowerModel):
                # Set Y limit
                ax1.set_ylim([-0.5, 14.5])
            else:
                # Set Y limit
                ax1.set_ylim([-0.5, 60.5])

            # Set Y label, grid and legend
            ax1.set_ylabel("% error", color='k')
            ax1.tick_params(axis='y', labelcolor='k')
            ax1.grid(True)
            ax1.legend()
            plt.tight_layout()  # Adjust subplot spacing

            # Create directory if it does not exit
            model_error_figures_dir = output_path
            if not os.path.exists(model_error_figures_dir):
                os.makedirs(model_error_figures_dir)

            # Ask the user for the figure name
            figure_save_file_name = input("Give me name to save this figure with "
                                          f"(path:{model_error_figures_dir}/<name>.pkl): ")

            # Save the figure
            with open(f"{model_error_figures_dir}/{figure_save_file_name}.pkl", 'wb') as file:
                pickle.dump(fig, file)

            # Plot the figure
            plt.show()

    def _grid_search_adaptative_train_zcu(self, train_df, grid_search_params):
        """Format the input observation dataframe and train each model on each
           of the observations.
        """

        # TODO: Not used on-chip. Handle this properly
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        # Format the dataframe
        [features_df, \
            top_power_labels_df, \
            bottom_power_labels_df, \
            time_labels_df] = processing.dataset_formating(train_df, power_rails="dual")

        # General variables
        outputs_dir = "/media/juan/HDD/tmp/model_error_figures_grid_search"
        total_json_dict = {}
        total_json_dict["iterations"] = {}

        #################
        ## Grid Search ##
        #################

        # Local variables used in the grid search algo
        param_names = {}
        param_combinations = {}
        total_combinations = 1

        # Iterate over each key ("train", "test") and get the parameters keys and values
        for group, group_params in grid_search_params.items():

            # Create a list with the parameters names
            param_names[group] = list(group_params.keys())
            # Create a list wiht the parameters values
            param_values = list(group_params.values())
            # Create a dict with a key for train and test
            # and add to the corresponding one a list with al the posible combinations of its parameters values
            param_combinations[group] = list(itertools.product(*param_values))
            # Get the lenght of this list for calculating the total number of combinations
            total_combinations *= len(param_combinations[group])

        # Disk Utiliazation calculations
        # One iteration size in MiB
        iteration_size_mib = 73  #MiB (will change if the models parameters, the json generation or the figures change)
        # Total iteration size in GiB
        full_grid_search_size_gib = (total_combinations * iteration_size_mib) / (2**10)
        # Free disk space
        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir, exist_ok=True)
        disk_usage = shutil.disk_usage(outputs_dir)
        free_space_gib = disk_usage.free / (2**30)    # Available space in bytes
        # Grid Search disk utilization (%)
        grid_search_disk_utilization_percentage = (full_grid_search_size_gib / free_space_gib) * 100
        # User warning
        print("\nGrid Search runs to go: {}\n".format(total_combinations))
        print("It will generate {:.3f} GiB ({} MiB per run). Which is {:.2f}% of your available disk ({:.3f} GiB).".format(
            full_grid_search_size_gib,
            iteration_size_mib,
            grid_search_disk_utilization_percentage,
            free_space_gib))
        if full_grid_search_size_gib > 0.9 * free_space_gib:
            print("\nThis grid search would use too much disk ({:.2f}%). You better free up some space and try again. (╥﹏╥)".format(grid_search_disk_utilization_percentage))
            exit(1)
        else:
            keep_executing = input("Do you want to continue (y/n)? ").strip().lower()
        # When the space used by the grid search is not too much (and the user accepts)
        if keep_executing != 'y':
            exit(1)

        # Best models variables
        best_top_error = None
        best_top_error_index = None
        less_top_train = None
        less_top_train_index = None
        top_error_list = []
        top_train_obs_list = []

        best_bottom_error = None
        best_bottom_error_index = None
        less_bottom_train = None
        less_bottom_train_index = None
        bottom_error_list = []
        bottom_train_obs_list = []

        best_time_error = None
        best_time_error_index = None
        less_time_train = None
        less_time_train_index = None
        time_error_list = []
        time_train_obs_list = []

        actual_iteration = 0
        # Grid Search: iterate over the list of parameters combinations for train and test
        for train_params in param_combinations["train"]:
            for test_params in param_combinations["test"]:
                # Construct a dict with a different combinations of parameters each iteration
                param_set = {
                    "train": dict(zip(param_names["train"], train_params)),
                    "test": dict(zip(param_names["test"], test_params))
                }

                # Gather actual grid parameters
                actual_grid_search_params = {}

                actual_grid_search_params["output_dir"] = outputs_dir
                actual_grid_search_params["iteration"] = actual_iteration

                actual_grid_search_params["parameters"] = param_set
                actual_grid_search_params["outputs"] = {}
                actual_grid_search_params["outputs"]["top_model"] = {}
                actual_grid_search_params["outputs"]["bottom_model"] = {}
                actual_grid_search_params["outputs"]["time_model"] = {}

                # Pretty print the parameters
                print("-------------------")
                print("Iteration #{}\n".format(actual_grid_search_params["iteration"]))
                pretty_dict = json.dumps(actual_grid_search_params["parameters"], indent=4)
                print("Parameters:")
                print(pretty_dict)

                # Create new models each iteration
                self._top_power_model = models.PowerModel("PS")
                self._bottom_power_model = models.PowerModel("PL")
                self._time_model = models.TimeModel()

                # Train the Top Power model
                actual_grid_search_params["outputs"]["top_model"] = self._top_power_model.grid_search_train_batch(
                    features_df,
                    top_power_labels_df,
                    actual_grid_search_params
                )
                # Train the Bottom Power model
                actual_grid_search_params["outputs"]["bottom_model"] = self._bottom_power_model.grid_search_train_batch(
                    features_df,
                    bottom_power_labels_df,
                    actual_grid_search_params
                )

                # Train the Time model
                actual_grid_search_params["outputs"]["time_model"] = self._time_model.grid_search_train_batch(
                    features_df,
                    time_labels_df,
                    actual_grid_search_params
                )

                # Combine the parameters and outputs dictionaries
                combined_dict = {"parameters": actual_grid_search_params["parameters"], "models": actual_grid_search_params["outputs"]}

                # Save a JSON with the information of this iteration
                model_error_figures_dir = "{}/iter_{}".format(actual_grid_search_params["output_dir"], actual_grid_search_params["iteration"])
                if not os.path.exists(model_error_figures_dir):
                    os.makedirs(model_error_figures_dir, exist_ok=True)

                # Open the file in write mode
                with open("{}/parameters.json".format(model_error_figures_dir), "w") as file:
                    # Dump the dictionary to the file with indentation for pretty printing
                    json.dump(combined_dict, file, indent=4, cls=MyJSONEncoder)

                # Store the generated dictionary in the total dictionary
                total_json_dict["iterations"][str(actual_iteration)] = combined_dict

                # Add a dict for storing error and train position (is done futher down) (We put a 0 to have the keys in that order)
                total_json_dict["iterations"][str(actual_iteration)]["models"]["top_model"]["positions"] = {}
                total_json_dict["iterations"][str(actual_iteration)]["models"]["top_model"]["positions"]["best_error"] = 0
                total_json_dict["iterations"][str(actual_iteration)]["models"]["top_model"]["positions"]["less_train"] = 0
                total_json_dict["iterations"][str(actual_iteration)]["models"]["bottom_model"]["positions"] = {}
                total_json_dict["iterations"][str(actual_iteration)]["models"]["bottom_model"]["positions"]["best_error"] = 0
                total_json_dict["iterations"][str(actual_iteration)]["models"]["bottom_model"]["positions"]["less_train"] = 0
                total_json_dict["iterations"][str(actual_iteration)]["models"]["time_model"]["positions"] = {}
                total_json_dict["iterations"][str(actual_iteration)]["models"]["time_model"]["positions"]["best_error"] = 0
                total_json_dict["iterations"][str(actual_iteration)]["models"]["time_model"]["positions"]["less_train"] = 0

                # Store the error and the trained_obs values to sort them later
                top_error_list.append(actual_grid_search_params["outputs"]["top_model"]["adaptative"]["average_mape"])
                bottom_error_list.append(actual_grid_search_params["outputs"]["bottom_model"]["adaptative"]["average_mape"])
                time_error_list.append(actual_grid_search_params["outputs"]["time_model"]["adaptative"]["average_mape"])
                top_train_obs_list.append(actual_grid_search_params["outputs"]["top_model"]["adaptative"]["trained_observations"])
                bottom_train_obs_list.append(actual_grid_search_params["outputs"]["bottom_model"]["adaptative"]["trained_observations"])
                time_train_obs_list.append(actual_grid_search_params["outputs"]["time_model"]["adaptative"]["trained_observations"])

                # Generate best error and less training stages
                if best_top_error is None or actual_grid_search_params["outputs"]["top_model"]["adaptative"]["average_mape"] < best_top_error:
                    best_top_error = actual_grid_search_params["outputs"]["top_model"]["adaptative"]["average_mape"]
                    best_top_error_index = actual_iteration
                if best_bottom_error is None or actual_grid_search_params["outputs"]["bottom_model"]["adaptative"]["average_mape"] < best_bottom_error:
                    best_bottom_error = actual_grid_search_params["outputs"]["bottom_model"]["adaptative"]["average_mape"]
                    best_bottom_error_index = actual_iteration
                if best_time_error is None or actual_grid_search_params["outputs"]["time_model"]["adaptative"]["average_mape"] < best_time_error:
                    best_time_error = actual_grid_search_params["outputs"]["time_model"]["adaptative"]["average_mape"]
                    best_time_error_index = actual_iteration

                if less_top_train is None or actual_grid_search_params["outputs"]["top_model"]["adaptative"]["trained_observations"] < less_top_train:
                    less_top_train = actual_grid_search_params["outputs"]["top_model"]["adaptative"]["trained_observations"]
                    less_top_train_index = actual_iteration
                if less_bottom_train is None or actual_grid_search_params["outputs"]["bottom_model"]["adaptative"]["trained_observations"] < less_bottom_train:
                    less_bottom_train = actual_grid_search_params["outputs"]["bottom_model"]["adaptative"]["trained_observations"]
                    less_bottom_train_index = actual_iteration
                if less_time_train is None or actual_grid_search_params["outputs"]["time_model"]["adaptative"]["trained_observations"] < less_time_train:
                    less_time_train = actual_grid_search_params["outputs"]["time_model"]["adaptative"]["trained_observations"]
                    less_time_train_index = actual_iteration

                actual_iteration += 1

        print("best_top_error ({}): {}".format(best_top_error_index, best_top_error))
        print("best_bottom_error ({}): {}".format(best_bottom_error_index, best_bottom_error))
        print("best_time_error ({}): {}".format(best_time_error_index, best_time_error))
        print("less_top_train ({}): {}".format(less_top_train_index, less_top_train))
        print("less_bottom_train ({}): {}".format(less_bottom_train_index, less_bottom_train))
        print("less_time_train ({}): {}".format(less_time_train_index, less_time_train))

        # Sort the error and training obs and add the corresponding index to each iteration
        # Use enumerate to get (index, value) pairs and sort them by value
        top_error_sorted_indices = sorted(enumerate(top_error_list), key=lambda x: x[1])
        bottom_error_sorted_indices = sorted(enumerate(bottom_error_list), key=lambda x: x[1])
        time_error_sorted_indices = sorted(enumerate(time_error_list), key=lambda x: x[1])
        top_train_obs_sorted_indices = sorted(enumerate(top_train_obs_list), key=lambda x: x[1])
        bottom_train_obs_sorted_indices = sorted(enumerate(bottom_train_obs_list), key=lambda x: x[1])
        time_train_obs_sorted_indices = sorted(enumerate(time_train_obs_list), key=lambda x: x[1])
        # Extract the indices from the sorted list
        top_error_sorted_indices = [index for index, value in top_error_sorted_indices]
        bottom_error_sorted_indices = [index for index, value in bottom_error_sorted_indices]
        time_error_sorted_indices = [index for index, value in time_error_sorted_indices]
        top_train_obs_sorted_indices = [index for index, value in top_train_obs_sorted_indices]
        bottom_train_obs_sorted_indices = [index for index, value in bottom_train_obs_sorted_indices]
        time_train_obs_sorted_indices = [index for index, value in time_train_obs_sorted_indices]

        # Add the sorted position (in term of best error and less train obs) to the corresponding models
        for position, (top_error_index, bottom_error_index, time_error_index, top_train_obs_index, bottom_train_obs_index, time_train_obs_index) in enumerate(zip(top_error_sorted_indices, bottom_error_sorted_indices, time_error_sorted_indices, top_train_obs_sorted_indices, bottom_train_obs_sorted_indices, time_train_obs_sorted_indices)):
            total_json_dict["iterations"][str(top_error_index)]["models"]["top_model"]["positions"]["best_error"] = position
            total_json_dict["iterations"][str(top_train_obs_index)]["models"]["top_model"]["positions"]["less_train"] = position
            total_json_dict["iterations"][str(bottom_error_index)]["models"]["bottom_model"]["positions"]["best_error"] = position
            total_json_dict["iterations"][str(bottom_train_obs_index)]["models"]["bottom_model"]["positions"]["less_train"] = position
            total_json_dict["iterations"][str(time_error_index)]["models"]["time_model"]["positions"]["best_error"] = position
            total_json_dict["iterations"][str(time_train_obs_index)]["models"]["time_model"]["positions"]["less_train"] = position

        total_json_dict["best_models"] = {
            "top": {
                "best_error": {
                    "value": best_top_error,
                    "index": best_top_error_index
                },
                "less_train": {
                    "value": less_top_train,
                    "index": less_top_train_index
                }
            },
            "bottom": {
                "best_error": {
                    "value": best_bottom_error,
                    "index": best_bottom_error_index
                },
                "less_train": {
                    "value": less_bottom_train,
                    "index": less_bottom_train_index
                }
            },
            "time": {
                "best_error": {
                    "value": best_time_error,
                    "index": best_time_error_index
                },
                "less_train": {
                    "value": less_time_train,
                    "index": less_time_train_index
                }
            }
        }

        # Save all the grid_search info in a JSON file
        with open("{}/grid_search.json".format(outputs_dir), "w") as file:
            # Dump the dictionary to the file with indentation for pretty printing
            json.dump(total_json_dict, file, indent=4, cls=MyJSONEncoder)

    def _grid_search_adaptative_train_zcu_multiprocessing(self, train_df, grid_search_params):
        """Format the input observation dataframe and train each model on each
           of the observations.
        """

        # TODO: Not used on-chip. Handle this properly
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        # Format the dataframe
        [features_df, \
            top_power_labels_df, \
            bottom_power_labels_df, \
            time_labels_df] = processing.dataset_formating(train_df, power_rails="dual")

        # General variables
        outputs_dir = "/media/juan/HDD/tmp/model_error_figures_grid_search"
        total_json_dict = {}
        total_json_dict["iterations"] = {}

        #################
        ## Grid Search ##
        #################

        # Local variables used in the grid search algo
        param_names = {}
        param_combinations = {}
        total_combinations = 1

        # Iterate over each key ("train", "test") and get the parameters keys and values
        for group, group_params in grid_search_params.items():

            # Create a list with the parameters names
            param_names[group] = list(group_params.keys())
            # Create a list wiht the parameters values
            param_values = list(group_params.values())
            # Create a dict with a key for train and test
            # and add to the corresponding one a list with al the posible combinations of its parameters values
            param_combinations[group] = list(itertools.product(*param_values))
            # Get the lenght of this list for calculating the total number of combinations
            total_combinations *= len(param_combinations[group])

        # Disk Utiliazation calculations
        # One iteration size in MiB
        iteration_size_mib = 73  #MiB (will change if the models parameters, the json generation or the figures change)
        # Total iteration size in GiB
        full_grid_search_size_gib = (total_combinations * iteration_size_mib) / (2**10)
        # Free disk space
        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir, exist_ok=True)
        disk_usage = shutil.disk_usage(outputs_dir)
        free_space_gib = disk_usage.free / (2**30)    # Available space in bytes
        # Grid Search disk utilization (%)
        grid_search_disk_utilization_percentage = (full_grid_search_size_gib / free_space_gib) * 100
        # User warning
        print("\nGrid Search runs to go: {}\n".format(total_combinations))
        print("It will generate {:.3f} GiB ({} MiB per run). Which is {:.2f}% of your available disk ({:.3f} GiB).".format(
            full_grid_search_size_gib,
            iteration_size_mib,
            grid_search_disk_utilization_percentage,
            free_space_gib))
        if full_grid_search_size_gib > 0.9 * free_space_gib:
            print("\nThis grid search would use too much disk ({:.2f}%). You better free up some space and try again. (╥﹏╥)".format(grid_search_disk_utilization_percentage))
            exit(1)
        else:
            keep_executing = input("Do you want to continue (y/n)? ").strip().lower()
        # When the space used by the grid search is not too much (and the user accepts)
        if keep_executing != 'y':
            exit(1)

        # Best models variables
        best_top_error = None
        best_top_error_index = None
        less_top_train = None
        less_top_train_index = None
        top_error_list = []
        top_train_obs_list = []

        best_bottom_error = None
        best_bottom_error_index = None
        less_bottom_train = None
        less_bottom_train_index = None
        bottom_error_list = []
        bottom_train_obs_list = []

        best_time_error = None
        best_time_error_index = None
        less_time_train = None
        less_time_train_index = None
        time_error_list = []
        time_train_obs_list = []

        actual_grid_search_params = []
        results1 = []
        results2 = []
        results3 = []

        # Create a pool of processes
        pool = multiprocessing.Pool(processes=6)

        actual_iteration = 0
        # Grid Search: iterate over the list of parameters combinations for train and test
        for train_params in param_combinations["train"]:
            for test_params in param_combinations["test"]:
                # Construct a dict with a different combinations of parameters each iteration
                param_set = {
                    "train": dict(zip(param_names["train"], train_params)),
                    "test": dict(zip(param_names["test"], test_params))
                }

                print("Iteration:", actual_iteration)
                # Gather actual grid parameters
                actual_grid_search_params.append({})

                actual_grid_search_params[actual_iteration]["output_dir"] = outputs_dir
                actual_grid_search_params[actual_iteration]["iteration"] = actual_iteration

                actual_grid_search_params[actual_iteration]["parameters"] = param_set
                actual_grid_search_params[actual_iteration]["outputs"] = {}
                actual_grid_search_params[actual_iteration]["outputs"]["top_model"] = {}
                actual_grid_search_params[actual_iteration]["outputs"]["bottom_model"] = {}
                actual_grid_search_params[actual_iteration]["outputs"]["time_model"] = {}

                # Pretty print the parameters
                # print("-------------------")
                # print("Iteration #{}\n".format(actual_grid_search_params["iteration"]))
                # pretty_dict = json.dumps(actual_grid_search_params["parameters"], indent=4)
                # print("Parameters:")
                # print(pretty_dict)

                ##############
                ## Procesos ##
                ##############

                # Create new models each iteration
                tmp_top_model = models.PowerModel("PS")
                tmp_bottom_model = models.PowerModel("PL")
                tmp_time_model = models.TimeModel()

                results1.append(pool.map_async(tmp_top_model.grid_search_train_batch_multiprocessing, [(features_df, top_power_labels_df, actual_grid_search_params[actual_iteration]),]))
                results2.append(pool.map_async(tmp_bottom_model.grid_search_train_batch_multiprocessing, [(features_df, bottom_power_labels_df, actual_grid_search_params[actual_iteration]),]))
                results3.append(pool.map_async(tmp_time_model.grid_search_train_batch_multiprocessing, [(features_df, time_labels_df, actual_grid_search_params[actual_iteration]),]))

                actual_iteration += 1

        #print(len(actual_grid_search_params), len(results1), len(results2))

        for i in range(len(actual_grid_search_params)):


            #print(len(actual_grid_search_params), len(results1), len(results2))
            print("[Iteration #{}] Waiting results.".format(i))
            results1[i].wait()
            results2[i].wait()
            results3[i].wait()
            # We access the first element bc the .get() returns a list... :(
            actual_grid_search_params[i]["outputs"]["top_model"] = results1[i].get()[0]
            #print(results1[i].get())
            #print(type(results1[i].get()))
            actual_grid_search_params[i]["outputs"]["bottom_model"] = results2[i].get()[0]
            actual_grid_search_params[i]["outputs"]["time_model"] = results3[i].get()[0]
            #print(actual_grid_search_params[i]["outputs"]["top_model"])
            #print(type(actual_grid_search_params[i]["outputs"]["top_model"]))
            # Combine the parameters and outputs dictionaries
            combined_dict = {"parameters": actual_grid_search_params[i]["parameters"], "models": actual_grid_search_params[i]["outputs"]}

            # Save a JSON with the information of this iteration
            model_error_figures_dir = "{}/iter_{}".format(actual_grid_search_params[i]["output_dir"], actual_grid_search_params[i]["iteration"])
            if not os.path.exists(model_error_figures_dir):
                os.makedirs(model_error_figures_dir, exist_ok=True)

            # Open the file in write mode
            with open("{}/parameters.json".format(model_error_figures_dir), "w") as file:
                # Dump the dictionary to the file with indentation for pretty printing
                json.dump(combined_dict, file, indent=4, cls=MyJSONEncoder)

            # Store the generated dictionary in the total dictionary
            total_json_dict["iterations"][str(i)] = combined_dict

            # Add a dict for storing error and train position (is done futher down) (We put a 0 to have the keys in that order)
            total_json_dict["iterations"][str(i)]["models"]["top_model"]["positions"] = {}
            total_json_dict["iterations"][str(i)]["models"]["top_model"]["positions"]["best_error"] = 0
            total_json_dict["iterations"][str(i)]["models"]["top_model"]["positions"]["less_train"] = 0
            total_json_dict["iterations"][str(i)]["models"]["bottom_model"]["positions"] = {}
            total_json_dict["iterations"][str(i)]["models"]["bottom_model"]["positions"]["best_error"] = 0
            total_json_dict["iterations"][str(i)]["models"]["bottom_model"]["positions"]["less_train"] = 0
            total_json_dict["iterations"][str(i)]["models"]["time_model"]["positions"] = {}
            total_json_dict["iterations"][str(i)]["models"]["time_model"]["positions"]["best_error"] = 0
            total_json_dict["iterations"][str(i)]["models"]["time_model"]["positions"]["less_train"] = 0

            # Store the error and the trained_obs values to sort them later
            top_error_list.append(actual_grid_search_params[i]["outputs"]["top_model"]["adaptative"]["average_mape"])
            bottom_error_list.append(actual_grid_search_params[i]["outputs"]["bottom_model"]["adaptative"]["average_mape"])
            time_error_list.append(actual_grid_search_params[i]["outputs"]["time_model"]["adaptative"]["average_mape"])
            top_train_obs_list.append(actual_grid_search_params[i]["outputs"]["top_model"]["adaptative"]["trained_observations"])
            bottom_train_obs_list.append(actual_grid_search_params[i]["outputs"]["bottom_model"]["adaptative"]["trained_observations"])
            time_train_obs_list.append(actual_grid_search_params[i]["outputs"]["time_model"]["adaptative"]["trained_observations"])

            # Generate best error and less training stages
            if best_top_error is None or actual_grid_search_params[i]["outputs"]["top_model"]["adaptative"]["average_mape"] < best_top_error:
                best_top_error = actual_grid_search_params[i]["outputs"]["top_model"]["adaptative"]["average_mape"]
                best_top_error_index = i
            if best_bottom_error is None or actual_grid_search_params[i]["outputs"]["bottom_model"]["adaptative"]["average_mape"] < best_bottom_error:
                best_bottom_error = actual_grid_search_params[i]["outputs"]["bottom_model"]["adaptative"]["average_mape"]
                best_bottom_error_index = i
            if best_time_error is None or actual_grid_search_params[i]["outputs"]["time_model"]["adaptative"]["average_mape"] < best_time_error:
                best_time_error = actual_grid_search_params[i]["outputs"]["time_model"]["adaptative"]["average_mape"]
                best_time_error_index = i

            if less_top_train is None or actual_grid_search_params[i]["outputs"]["top_model"]["adaptative"]["trained_observations"] < less_top_train:
                less_top_train = actual_grid_search_params[i]["outputs"]["top_model"]["adaptative"]["trained_observations"]
                less_top_train_index = i
            if less_bottom_train is None or actual_grid_search_params[i]["outputs"]["bottom_model"]["adaptative"]["trained_observations"] < less_bottom_train:
                less_bottom_train = actual_grid_search_params[i]["outputs"]["bottom_model"]["adaptative"]["trained_observations"]
                less_bottom_train_index = i
            if less_time_train is None or actual_grid_search_params[i]["outputs"]["time_model"]["adaptative"]["trained_observations"] < less_time_train:
                less_time_train = actual_grid_search_params[i]["outputs"]["time_model"]["adaptative"]["trained_observations"]
                less_time_train_index = i

        pool.close()
        pool.join()

        print("best_top_error ({}): {}".format(best_top_error_index, best_top_error))
        print("best_bottom_error ({}): {}".format(best_bottom_error_index, best_bottom_error))
        print("best_time_error ({}): {}".format(best_time_error_index, best_time_error))
        print("less_top_train ({}): {}".format(less_top_train_index, less_top_train))
        print("less_bottom_train ({}): {}".format(less_bottom_train_index, less_bottom_train))
        print("less_time_train ({}): {}".format(less_time_train_index, less_time_train))

        # Sort the error and training obs and add the corresponding index to each iteration
        # Use enumerate to get (index, value) pairs and sort them by value
        top_error_sorted_indices = sorted(enumerate(top_error_list), key=lambda x: x[1])
        bottom_error_sorted_indices = sorted(enumerate(bottom_error_list), key=lambda x: x[1])
        time_error_sorted_indices = sorted(enumerate(time_error_list), key=lambda x: x[1])
        top_train_obs_sorted_indices = sorted(enumerate(top_train_obs_list), key=lambda x: x[1])
        bottom_train_obs_sorted_indices = sorted(enumerate(bottom_train_obs_list), key=lambda x: x[1])
        time_train_obs_sorted_indices = sorted(enumerate(time_train_obs_list), key=lambda x: x[1])
        # Extract the indices from the sorted list
        top_error_sorted_indices = [index for index, value in top_error_sorted_indices]
        bottom_error_sorted_indices = [index for index, value in bottom_error_sorted_indices]
        time_error_sorted_indices = [index for index, value in time_error_sorted_indices]
        top_train_obs_sorted_indices = [index for index, value in top_train_obs_sorted_indices]
        bottom_train_obs_sorted_indices = [index for index, value in bottom_train_obs_sorted_indices]
        time_train_obs_sorted_indices = [index for index, value in time_train_obs_sorted_indices]

        # Add the sorted position (in term of best error and less train obs) to the corresponding models
        for position, (top_error_index, bottom_error_index, time_error_index, top_train_obs_index, bottom_train_obs_index, time_train_obs_index) in enumerate(zip(top_error_sorted_indices, bottom_error_sorted_indices, time_error_sorted_indices, top_train_obs_sorted_indices, bottom_train_obs_sorted_indices, time_train_obs_sorted_indices)):
            total_json_dict["iterations"][str(top_error_index)]["models"]["top_model"]["positions"]["best_error"] = position
            total_json_dict["iterations"][str(top_train_obs_index)]["models"]["top_model"]["positions"]["less_train"] = position
            total_json_dict["iterations"][str(bottom_error_index)]["models"]["bottom_model"]["positions"]["best_error"] = position
            total_json_dict["iterations"][str(bottom_train_obs_index)]["models"]["bottom_model"]["positions"]["less_train"] = position
            total_json_dict["iterations"][str(time_error_index)]["models"]["time_model"]["positions"]["best_error"] = position
            total_json_dict["iterations"][str(time_train_obs_index)]["models"]["time_model"]["positions"]["less_train"] = position

        total_json_dict["best_models"] = {
            "top": {
                "best_error": {
                    "value": best_top_error,
                    "index": best_top_error_index
                },
                "less_train": {
                    "value": less_top_train,
                    "index": less_top_train_index
                }
            },
            "bottom": {
                "best_error": {
                    "value": best_bottom_error,
                    "index": best_bottom_error_index
                },
                "less_train": {
                    "value": less_bottom_train,
                    "index": less_bottom_train_index
                }
            },
            "time": {
                "best_error": {
                    "value": best_time_error,
                    "index": best_time_error_index
                },
                "less_train": {
                    "value": less_time_train,
                    "index": less_time_train_index
                }
            }
        }

        # Save all the grid_search info in a JSON file
        with open("{}/grid_search.json".format(outputs_dir), "w") as file:
            # Dump the dictionary to the file with indentation for pretty printing
            json.dump(total_json_dict, file, indent=4, cls=MyJSONEncoder)


if __name__ == "__main__":
    # TODO: Update this test
    import threading

    # Read observations file
    obs_df = processing.read_observations_from_file(
        "./observations.pkl"
    )

    # Sample train and test sets from the observations
    training_df, testing_df = processing.sample_observations(obs_df)

    # Initialize the models
    online_models = OnlineModels(power_rails="dual", lock=threading.Lock)

    # Old format
    training_df = training_df.drop(["Order"], axis=1)

    # Train and test
    online_models.train(training_df)

    train_metrics = online_models.get_metrics()
    print("Metrics Train: {} (top) | {} (bottom) | {} (time)".format(
        train_metrics[0],
        train_metrics[1],
        train_metrics[2]
    ))
    test_metrics = online_models.test(testing_df)
    print("Metrics Test: {} (top) | {} (bottom) | {} (time)".format(
        test_metrics[0],
        test_metrics[1],
        test_metrics[2]
    ))

    # Predict one (the last observation)
    last_obs_dict = testing_df.iloc[-1].drop(["Order"])

    print(type(last_obs_dict))
    print(last_obs_dict)

    last_obs_dict = last_obs_dict.to_dict()  # convert the observation to dict
    print(type(last_obs_dict))
    print(last_obs_dict)

    tp, bp, ti = online_models.predict_one(last_obs_dict)  # make prediction

    print("Top Power -> {} (Real) | {} (Prediction)".format(
        last_obs_dict["Top power"],
        tp)
    )
    print("Bottom Power -> {} (Real) | {} (Prediction)".format(
        last_obs_dict["Bottom power"],
        bp)
    )
    print("Time -> {} (Real) | {} (Prediction)".format(
        last_obs_dict["Time"],
        ti)
    )
