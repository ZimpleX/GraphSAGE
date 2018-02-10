import tensorflow as tf

import graphsage.models as models
import graphsage.layers as layers
from graphsage.aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator

import z_macro as z

flags = tf.app.flags
FLAGS = flags.FLAGS

class SupervisedGraphsage(models.SampleAndAggregate):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, num_classes,
            placeholders, features, adj, degrees,
            layer_infos, concat=True, aggregator_type="mean", 
            model_size="small", sigmoid_loss=False, identity_dim=0,
                **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''

        models.GeneralizedModel.__init__(self, **kwargs)

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from placeholders...
        self.inputs1 = placeholders["batch"]
        self.model_size = model_size
        self.adj_info = adj
        if identity_dim > 0:
           self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
           self.embeds = None
        if features is None: 
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        else:
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                # [z]: what does this mean?
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.degrees = degrees
        self.concat = concat
        self.num_classes = num_classes
        self.sigmoid_loss = sigmoid_loss
        # [z]: check where is dims used?
        # [z]: after next line, self.dims = 50
        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        # [z]: after next line, self.dims = [50,128,128]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        # [z]: check this!
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        
        # [z]: build in the init function defines the rule of calc the variables.
        #   However, the actual values are in fact calculated in supervised_train.py/train()
        self.build()


    def build(self):
        #####################
        # [z]: SAMPLING     #
        #   for all layers  #
        #####################
        # [z]: samples1: [array of 512, array of 5120, array of 128000]
        # [Z]: should get the adj matrix connecting the two layers
        samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos)      # [z]: check neigh_sampler.py
        z.debug_vars['supervised_models/build/samples1'] = samples1
        # [z]: num_samples = [25,10]
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        #import pdb; pdb.set_trace()
        # [z]: self.aggregate is in superclass
        #####################
        # [z]: FORWARD PROP #
        #####################
        # [z]: self.features is the input features for each node (a length 50 vector)
        # [z]: self.dims is the number of input features for each conv layer
        self.outputs1, self.aggregators = self.aggregate(samples1, [self.features], self.dims, num_samples,
                support_sizes1, concat=self.concat, model_size=self.model_size)
        dim_mult = 2 if self.concat else 1
        #####################
        # [z]: OUPTUT LAYER #
        #####################
        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)

        dim_mult = 2 if self.concat else 1
        # [z]: final output, predict class
        # [z]: self.num_classes = 121
        #   self.dims = [50,128,128]
        #   dim_mult = 2
        #   self.num_classes = 121
        self.node_pred = layers.Dense(dim_mult*self.dims[-1], self.num_classes, 
                dropout=self.placeholders['dropout'],
                act=lambda x : x)
        # TF graph management
        # [z]: self.node_preds is R^{?x121}, where 121 is the number of classes
        self.node_preds = self.node_pred(self.outputs1)

        #####################
        # [z]: BACK PROP    #
        #####################
        self._loss()
        # [z]: start to backprop?
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        # [z]: update param by gradient?
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        self.preds = self.predict()

    def _loss(self):
        # Weight decay loss
        # [z]: see this: https://stats.stackexchange.com/questions/29130/difference-between-neural-net-weight-decay-and-learning-rate
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
       
        # classification loss
        if self.sigmoid_loss:
            self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.placeholders['labels']))
        else:
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.placeholders['labels']))
        tf.summary.scalar('loss', self.loss)

    def predict(self):
        if self.sigmoid_loss:
            return tf.nn.sigmoid(self.node_preds)
        else:
            return tf.nn.softmax(self.node_preds)
