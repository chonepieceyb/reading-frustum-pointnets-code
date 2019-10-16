center_delta, end_points = get_center_regression_net( \
        point_cloud, one_hot_vec,
        is_training, bn_decay, end_points)#(B,3)
    point_cloud_xyz = tf.slice(point_cloud, [0, 0, 0], [-1, -1, 3])  # BxNx3
    point_cloud_differ=point_cloud_xyz-tf.expand_dims(center_delta, 1)
    point_cloud_differ_x=tf.slice(point_cloud_differ, [0, 0, 0], [-1, -1, 1])
    point_cloud_differ_y = tf.slice(point_cloud_differ, [0, 0, 0], [-1, -1, 2])
    point_cloud_differ_z = tf.slice(point_cloud_differ, [0, 0, 0], [-1, -1, 3])
    point_cloud_H=point_cloud_differ_x*point_cloud_differ_x+point_cloud_differ_y*point_cloud_differ_y\
                    +point_cloud_differ_z*point_cloud_differ_z
    point_cloud_H=tf.sqrt(point_cloud_H)

    point_cloud = tf_util.conv2d(point_cloud_H, 2, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    point_cloud = tf_util.conv2d(point_cloud, 2, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    point_cloud = tf_util.conv2d(point_cloud, 3, [1, 1],
                                padding='VALID', stride=[1, 1],
                                bn=True, is_training=is_training,
                                scope='conv3', bn_decay=bn_decay)

    point_cloud_features = tf.slice(point_cloud, [0, 0, 3], [-1, -1, -1])
    point_cloud=tf.concat([point_cloud, point_cloud_features], axis=-1)
