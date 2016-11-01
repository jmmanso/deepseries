import tensorflow as tf
import numpy as np


class cost_functions:
    @classmethod
    def classification(self,
        output_logits_tensor,
        target_tensor,
        norm_coeff):
        """ Returns softmax X-entropy error for a pair of
        model/truth tensors, with sample weights """
        # Sum the weighted loss vector into a scalar
        result = tf.reduce_sum(
                    # Hadamard product between loss and norm coeff vectors
                    tf.mul(
                        # Get the (Nchannels,) vector with the scalar
                        # errors for each Nsequence sample.
                        tf.nn.softmax_cross_entropy_with_logits(
                            output_logits_tensor, target_tensor),
                        # norm_coeff are nested, we flatten them here
                        tf.squeeze(norm_coeff)
                        )
                    )
        return result

    @classmethod
    def regression(self,
        output_logits_tensor,
        target_tensor,
        norm_coeff):
        """ Returns l2 error for a pair of model/truth tensors,
        with sample weights """
        # Sum the weighted loss vector into a scalar
        result = tf.reduce_sum(
                    # Hadamard product between loss and norm coeff vectors
                    tf.mul(
                        # Get the (Nchannels,) vector with the scalar
                        # errors for each Nsequence sample.
                        # This l2 tensor is rank 2:
                        tf.sub(
                            tf.tanh(output_logits_tensor), target_tensor
                            )**2,
                        norm_coeff),
                    # `1` to sum over channel axis
                    1)/2.0
        return result

    @classmethod
    def combined(self,
        output_logits_tensor,
        target_tensor,
        y_categorical_indices_container,
        y_continuous_indices,
        norm_coeff):
        """ Computes loss between model/truth tensors, applying
        softmax X-entropy for categorical variables and l2 for continuous ones.


        Parameters
        ----------
        output_logits_tensor : tensor, shape = (Nsequences, Nchannels_ouput)
                Output logit Nchannel-vector for every sequence.
        target_tensor: tensor, shape = (Nsequences, Nchannels_ouput)
                Target (truth) Nchannel-vector for every sequence.
        y_categorical_indices_container: list
                List containing lists, each one with the indices of a
                categorical channel group. If there are no categorical
                groups, this parent list can be empty.
        y_continuous_indices: list
                Contains the indices of continuous variables. Unlike
                y_categorical_indices_container, it is not nested. Can
                be empty if there are no continuous variables.
        norm_coeff: tensor, shape = (Nsequences, 1)
                Numerical coefficients to weigh the contribution of each
                sequence to the total loss
        Returns
        -------
        cost : float
                combined batch loss
        """
        # If there are any continuous indices, then
        if len(y_continuous_indices)>0:
            # get the l2 loss
            regr_cost = self.regression(
                            tf.transpose(
                                # fetch only relevant indices
                                tf.gather(
                                    tf.transpose(output_logits_tensor), y_continuous_indices)
                                ),
                            #
                            tf.transpose(
                                tf.gather(
                                    tf.transpose(target_tensor),
                                    y_continuous_indices)
                                ),
                            norm_coeff)
        else:
            regr_cost = 0
        # If there are any categorical groups, then
        if len(y_categorical_indices_container)>0:
            # get the loss for this categorical group, and add it
            # to the classifcation loss container
            class_cost_container = []
            for y_categorical_indices in y_categorical_indices_container:
                class_cost_container.append(self.classification(\
                    # fetch only indices from theis group
                    #(trying to do: output_logits_tensor[:, categorical_indices])
                    tf.transpose(
                        tf.gather(
                            tf.transpose(output_logits_tensor),
                            y_categorical_indices)
                        ),
                    tf.transpose(
                        tf.gather(
                            tf.transpose(target_tensor),
                            y_categorical_indices)
                        ),
                    norm_coeff))
            # once done with all categorical groups, sum their losses
            class_cost = sum(class_cost_container)
        else:
            class_cost = 0

        return class_cost + regr_cost

class prediction_functions:
    @classmethod
    def classification(self, output_logits_tensor):
        """ Applies softmax to a tensor """
        return tf.nn.softmax(output_logits_tensor)

    @classmethod
    def regression(self, output_logits_tensor):
        """ Applies tanh to a tensor."""
        return tf.tanh(output_logits_tensor)

    @classmethod
    def combined(self,
        output_logits_tensor,
        y_categorical_indices_container,
        y_continuous_indices):
        """ Projects an output tensor into continuous
        and categorical (softmax) predictions.

        Parameters
        ----------
        output_logits_tensor : tensor, shape = (Nsequences, Nchannels_ouput)
                Output logit Nchannel-vector for every sequence.
        y_categorical_indices_container: list
                List containing lists, each one with the indices of a
                categorical channel group. If there are no categorical
                groups, this parent list can be empty.
        y_continuous_indices: list
                Contains the indices of continuous variables. Unlike
                y_categorical_indices_container, it is not nested. Can
                be empty if there are no continuous variables.
        Returns
        -------
        ordered_pred_tensor : tensor, shape = (Nsequences, Nchannels_ouput)
            The predicted outputs. Column i corresponds to the index of
            value i in the supplied index lists.  """
        # List to contain the prediction tensors for every group
        # (categorical groups + continuous)
        pred_container = []
        # List to contain the index lists for every group, following the
        # same order as pred_container
        id_container = []
        # If there are any categorical groups, then
        if len(y_categorical_indices_container)>0:
            # get the prediction tensor for this categorical group
            for y_categorical_indices in y_categorical_indices_container:
                class_pred = self.classification(\
                        tf.transpose(
                            # Fetch only relevant indices
                            tf.gather(
                                tf.transpose(output_logits_tensor), y_categorical_indices)
                                )
                            )
                pred_container.append(class_pred)
                id_container.append(y_categorical_indices)
        # If there are any continuous indices, then
        if len(y_continuous_indices)>0:
            # get the prediction tensor
            regr_pred = self.regression(\
                            tf.transpose(
                                tf.gather(
                                    tf.transpose(output_logits_tensor), y_continuous_indices)
                                    )
                                )
            # Append to containers
            pred_container.append(regr_pred)
            id_container.append(y_continuous_indices)

        # Flatten index container to list, preserving container order
        id_list = [val for sublist in id_container for val in sublist]
        # do a similar thing with prediction tensors: concatenate them
        # into a single tensor
        pred_tensor = tf.concat(1, pred_container)
        # reshuffle the tensor columns to follow the order dictated by
        # the values of the indices in the list
        ordered_pred_tensor = tf.transpose(tf.gather(tf.transpose(pred_tensor), id_list))
        return ordered_pred_tensor

def softmax2onehot(array):
    """ Transforms softmax outputs of
    shape (Nsequences, sequence_length, Nchannels)
    into one-hot arrays of the same shape. Essentially,
    with 1's at the highest prob channel, 0's elsewhere """
    ids = np.argmax(array, 2)
    new_array = np.zeros_like(array, dtype=np.float32)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            new_array[i, j, ids[i, j]] = 1
    #
    return new_array
