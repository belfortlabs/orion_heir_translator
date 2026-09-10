def ciphertext_chunks(inputs, slots):
    """Flatten forward arguments into consecutive ciphertext-sized chunks."""
    inputs = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
    return [chunk for value in inputs for chunk in value.flatten().split(slots)]
