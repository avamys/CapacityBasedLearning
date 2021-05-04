class TBLogger():
    def __init__(self, writer):
        self.writer = writer

    def log_model_params(self, model, epoch):
        ''' Writes number of buds and values of lipschitz constants for layers
            in each neuron bud to TensorBoard writer 
        '''
        # Level 0
        for layer_id, layer in enumerate(model.layerlist):
            # Log number of buds in each layer in level 0
            self.writer.add_scalar(
                f'buds_layer{layer_id}', 
                len(layer.buds), 
                epoch)

            # Log value of best lipschitz constants in level 0
            lips = layer.get_lipschitz_constant()
            if lips is not None:
                keys = [str(i) for i in range(lips.shape[0])]
                self.writer.add_scalars(
                    f'lipschitz_layer{layer_id}', 
                    dict(zip(keys, lips)), 
                    epoch)

        # Log total number of buds
        log_model_n_buds, log_model_lipschitz = model.get_n_buds()
        self.writer.add_scalar('total_n_buds', log_model_n_buds, epoch)

        # Log values of lipschitz constants in buds
        for lipschitz_key in log_model_lipschitz:
            self.writer.add_scalars(
                lipschitz_key, 
                log_model_lipschitz[lipschitz_key], 
                epoch)