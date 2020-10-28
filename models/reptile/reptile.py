import tensorflow as tf

from models.maml.maml import ModelAgnosticMetaLearningModel
from utils import combine_first_two_axes

"""This model still has bugs. """
class Reptile(ModelAgnosticMetaLearningModel):
    def __init__(
            self,
            database,
            network_cls,
            n,
            k_ml,
            k_val_ml,
            k_val,
            k_val_val,
            k_test,
            k_val_test,
            meta_batch_size,
            num_steps_ml,
            lr_inner_ml,
            num_steps_validation,
            save_after_iterations,
            meta_learning_rate,
            report_validation_frequency,
            log_train_images_after_iteration,  # Set to -1 if you do not want to log train images.
            num_tasks_val,  # Make sure the validation pick this many tasks.
            val_seed=-1,  # The seed for validation dataset. -1 means change the samples for each report.
            clip_gradients=False,
            experiment_name=None,
            val_test_batch_norm_momentum=0.0,
            val_database=None,
            test_database=None,
    ):
        super(Reptile, self).__init__(
            database,
            network_cls,
            n,
            k_ml,
            k_val_ml,
            k_val,
            k_val_val,
            k_test,
            k_val_test,
            meta_batch_size,
            num_steps_ml,
            lr_inner_ml,
            num_steps_validation,
            save_after_iterations,
            meta_learning_rate,
            report_validation_frequency,
            log_train_images_after_iteration,  # Set to -1 if you do not want to log train images.
            num_tasks_val,  # Make sure the validation pick this many tasks.
            val_seed=val_seed,  # The seed for validation dataset. -1 means change the samples for each report.
            clip_gradients=clip_gradients,
            experiment_name=experiment_name,
            val_test_batch_norm_momentum=val_test_batch_norm_momentum,
            val_database=val_database,
            test_database=test_database,
        )
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=meta_learning_rate, beta_1=0)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=meta_learning_rate)

    @tf.function
    def meta_train_loop(self, train_ds, val_ds, train_labels, val_labels):
        tasks_final_losses = list()
        tasks_final_accs = list()

        weight_updates = dict()

        for i in range(self.meta_batch_size):
            task_train_set = combine_first_two_axes(train_ds[i, ...])
            task_labels = train_labels[i, ...]
            updated_model = self.inner_train_loop(task_train_set, task_labels)

            # TODO evaluate it on task validation for training loss?
            updated_model_logits = updated_model(task_train_set, training=True)
            loss = self.outer_loss(task_labels, updated_model_logits)
            tasks_final_losses.append(loss)

            predicted_class_labels = self.predict_class_labels_from_logits(updated_model_logits)
            real_labels = self.convert_labels_to_real_labels(task_labels)

            acc = tf.reduce_mean(tf.cast(tf.equal(predicted_class_labels, real_labels), tf.float32))
            tasks_final_accs.append(acc)

            for variable in self.model.trainable_variables:
                references = self.extract_variable_reference_from_variable_name(variable.name)
                layer_names = references[:-1]
                attr = references[-1]

                model_layer = self.model
                updated_model_layer = updated_model
                for layer_name in layer_names:
                    model_layer = model_layer.get_layer(layer_name)
                    updated_model_layer = updated_model_layer.get_layer(layer_name)

                update_direction = updated_model_layer.__dict__[attr] - model_layer.__dict__[attr]
                weight_updates[variable.name] = weight_updates.get(
                    variable.name, tf.zeros_like(update_direction)
                ) + update_direction / self.meta_batch_size

        gradients = []

        for variable in self.model.trainable_variables:
            if variable.name in weight_updates:
                gradients.append(-1 * weight_updates[variable.name])
                # references = self.extract_variable_reference_from_variable_name(variable.name)
                # layer_names = references[:-1]
                # attr = references[-1]
                #
                # model_layer = self.model
                # for layer_name in layer_names:
                #     model_layer = model_layer.get_layer(layer_name)
                #
                # model_layer.__dict__[attr].assign(variable + weight_updates[variable.name])

                # variable.assign(variable + self.meta_learning_rate * weight_updates[variable.name])

        final_acc = tf.reduce_mean(tasks_final_accs)
        final_loss = tf.reduce_mean(tasks_final_losses)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return final_acc, final_loss
