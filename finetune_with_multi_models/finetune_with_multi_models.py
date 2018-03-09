#load model with name_scope or op name


def load_pretrained_scope_weight(file_name, include_scope, session):
    #file_name:(type:string)*.ckpt-*
    #include_scope:(type:list)scope_name or para name list
    #write in the place where saver.restore used

    #if para name list use this
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(include=include_scope)
    #if scope_name use this
    #variables_to_restore = tf.contrib.framework.get_variables_to_restore(include=[include_scope])
    init_fn = tf.contrib.framework.assign_from_checkpoint_fn(file_name, variables_to_restore,ignore_missing_vars=True)
    init_fn(session)

#trainable=None：train all op
#trainable=tf.trainable_variables(scope_name)：train only  scope_name


def build_cnn_6_with_skip_connection_classifier(image_batch, num_class,training):
    scope_name = "plant_seedings_cnn_6_with_skip_connection_classifier" 
    with tf.variable_scope(scope_name): 
        flatten = model.bn_cnn.build_bn_cnn_6_with_skip_connection(image_batch, training) 
        linear = tf.layers.dense(flatten, num_class, name='fc') 
        logits = tf.nn.softmax(linear, name='softmax') 
     return linear, logits, tf.trainable_variables(scope_name) 


def build_train_op(self, loss, global_step, trainable=None,optimizer=tf.train.AdamOptimizer):
#optimizer_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
#optimizer_op=optimizer.compute_gradients+optimizer.apply_gradients
    optimizer = optimizer(learning_rate=learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    if update_ops:
        with tf.control_dependencies([tf.group(*update_ops)]):
            grad_var = optimizer.compute_gradients(loss, var_list=trainable)
            optimizer_op=optimizer.apply_gradients(grad_var,global_step=global_step)
            return optimizer_op
    else:
        grad_var = optimizer.compute_gradients(loss, var_list=trainable)
        optimizer_op=optimizer.apply_gradients(grad_var,global_step=global_step)
        return optimizer_op
    return

	