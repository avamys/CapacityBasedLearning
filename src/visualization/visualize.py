import wandb
from src.models.network import NeuronBud, BuddingLayer

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
            if isinstance(layer, BuddingLayer):
                self.writer.add_scalar(
                    f'buds_layer{layer_id}', 
                    len(layer.buds), 
                    epoch)
                wandb.log({f'buds_layer{layer_id}': len(layer.buds)}, step=epoch)

                # Log value of best lipschitz constants in level 0
                lips = layer.get_lipschitz_constant()
                if lips is not None:
                    keys = [str(i) for i in range(lips.shape[0])]
                    self.writer.add_scalars(
                        f'lipschitz_layer{layer_id}', 
                        dict(zip(keys, lips)), 
                        epoch)

        # Log total number of buds
        self.writer.add_scalar('total_n_buds', NeuronBud.counter, epoch)
        wandb.log({'total_n_buds': NeuronBud.counter}, step=epoch)

        # Log values of lipschitz constants and number of buds in every layer
        log_model_lipschitz, log_model_buds = model.get_model_params()
        wandb.log(log_model_lipschitz, step=epoch)
        wandb.log(log_model_buds, step=epoch)

        for lipschitz_key in log_model_lipschitz:
            self.writer.add_scalars(
                lipschitz_key,
                log_model_lipschitz[lipschitz_key], 
                epoch)
            # wandb.log({lipschitz_key: log_model_lipschitz[lipschitz_key]})

        for bud_key in log_model_buds:
            self.writer.add_scalar(
                bud_key,
                log_model_buds[bud_key],
                epoch)
            # wandb.log({bud_key: log_model_buds[bud_key]})