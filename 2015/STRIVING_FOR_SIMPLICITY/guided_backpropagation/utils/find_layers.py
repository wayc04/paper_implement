# this is a function that replace all the layers of a certain type in a model with a new layer
def replace_all_layer_type_recursive(model, old_layer_type, new_layer):
    for name, layer in model._modules.items():
        # if the layer is an instance of the old layer type, replace it with the new layer
        if isinstance(layer, old_layer_type):
            model._modules[name] = new_layer

        replace_all_layer_type_recursive(layer, old_layer_type, new_layer)
